import os
import streamlit as st
import streamlit.components.v1 as components
import uuid
import logging
import secrets
from typing import List, Annotated, TypedDict, Dict
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, add_messages, END, START
from langgraph.checkpoint.postgres import PostgresSaver
import psycopg
from prompts import PERSONA_RAFAEL, PERSONA_CLARA, PERSONA_LUIZ, EVALUATION_SESSION_1, EVALUATION_SESSION_2

# --- 1. CONFIGURAÇÃO INICIAL E LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
st.set_page_config(page_title="Simulador de Terapia (Project Match)", page_icon="⚕️")
load_dotenv()

# --- 2. CONSTANTES E VALIDAÇÕES INICIAIS ---
END_SESSION_CODE = "H7Y4K9P2R1T6X3Z0V8B5N7M3G"
EVALUATION_METADATA_KEY = "is_evaluation"
MIN_PROMPT_LENGTH = 10
BRAZIL_TZ = timezone(timedelta(hours=-3))
CLOCK_HTML = """<style> .digital-clock { background-color: #e1e5eb; border: 2px solid #c9ced4; border-radius: 5px; padding: 8px; font-family: sans-serif; color: #0d1a33; font-size: 1.75rem; font-weight: bold; text-align: center; letter-spacing: 2px; } </style><script> function updateClock() { var now = new Date(); var h = now.getHours().toString().padStart(2, '0'); var m = now.getMinutes().toString().padStart(2, '0'); var s = now.getSeconds().toString().padStart(2, '0'); document.getElementById('clock').innerText = h + ':' + m + ':' + s; } setInterval(updateClock, 1000); setTimeout(updateClock, 1); </script><div id="clock" class="digital-clock"></div>"""

# --- WHITELIST DE USUÁRIOS AUTORIZADOS ---
ALLOWED_USER_IDS_RAW = os.getenv("ALLOWED_USER_IDS", "")
ALLOWED_USER_IDS = set(uid.strip() for uid in ALLOWED_USER_IDS_RAW.split(",") if uid.strip())

if not ALLOWED_USER_IDS:
    st.error("⚠️ ALLOWED_USER_IDS não configurado no arquivo .env! Aplicação bloqueada.")
    st.stop()

logger.info(f"✅ Whitelist carregada com {len(ALLOWED_USER_IDS)} usuários autorizados")

# Validação de API Keys
api_key = os.getenv("OPENAI_API_KEY")
model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
if not api_key or not model:
    st.error("⚠️ OPENAI_API_KEY ou OPENAI_MODEL não encontrados! Configure no arquivo .env")
    st.stop()

# --- 3. ESTRUTURAS DE DADOS E DEFINIÇÕES GLOBAIS ---
# ORDEM FIXA: Clara → Rafael → Luiz
PERSONAS_DATA = [
    {"name": "Clara", "prompt": PERSONA_CLARA, "order": 1},
    {"name": "Rafael", "prompt": PERSONA_RAFAEL, "order": 2},
    {"name": "Luiz", "prompt": PERSONA_LUIZ, "order": 3},
]

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    current_session: int
    session_1_end_index: int
    patient_prompt: str
    persona_name: str

# --- 4. FUNÇÕES HELPER GLOBAIS ---
def is_valid_uuid(val: str) -> bool:
    """Valida se uma string é um UUID válido."""
    try:
        uuid.UUID(str(val))
        return True
    except (ValueError, AttributeError, TypeError):
        return False

def filter_messages(messages: List[BaseMessage]) -> List[BaseMessage]:
    return [msg for msg in messages if not (isinstance(msg, HumanMessage) and END_SESSION_CODE in msg.content) and not (isinstance(msg, AIMessage) and msg.response_metadata.get(EVALUATION_METADATA_KEY))]

def create_transcript(messages: List[BaseMessage]) -> str:
    return "\n".join([f"{'Terapeuta' if isinstance(msg, HumanMessage) else 'Paciente'}: {msg.content}" for msg in messages])

def route_entry_point(state: AgentState) -> str:
    last_message = state["messages"][-1] if state["messages"] else None
    if isinstance(last_message, HumanMessage) and END_SESSION_CODE in last_message.content:
        current_session = state.get("current_session", 1)
        if current_session == 1: return "evaluate_session_1"
        elif current_session == 2: return "evaluate_session_2"
    return "patient_node"

def get_next_persona(current_persona_name: str) -> Dict:
    """Retorna o próximo paciente na ordem fixa: Clara → Rafael → Luiz → Clara..."""
    current_persona = next((p for p in PERSONAS_DATA if p["name"] == current_persona_name), None)
    
    if not current_persona:
        # Se não encontrar, retorna Clara (primeira da lista)
        return PERSONAS_DATA[0]
    
    current_order = current_persona["order"]
    
    # Busca o próximo na ordem
    next_persona = next((p for p in PERSONAS_DATA if p["order"] == current_order + 1), None)
    
    # Se não houver próximo, volta para o primeiro (Clara)
    if not next_persona:
        next_persona = PERSONAS_DATA[0]
    
    return next_persona

# --- 5. FUNÇÕES CACHEADAS PARA RECURSOS CAROS (DB, LLMs e Grafo) ---

@st.cache_resource
def get_db_connection():
    logger.info("Criando conexão com o banco de dados (executado apenas uma vez)...")
    try:
        conn = psycopg.connect(
            host=os.getenv("POSTGRES_HOST"),
            port=os.getenv("POSTGRES_PORT"),
            user=os.getenv("POSTGRES_USER"),
            password=os.getenv("POSTGRES_PASSWORD"),
            dbname=os.getenv("POSTGRES_DB"),
            sslmode='require'
        )
        
        # Criar/Atualizar tabela para metadados de sessão
        with conn.cursor() as cur:
            # Primeiro, cria a tabela se não existir
            cur.execute("""
                CREATE TABLE IF NOT EXISTS session_metadata (
                    thread_id TEXT PRIMARY KEY,
                    persona_name TEXT NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Adiciona a coluna last_accessed se não existir
            cur.execute("""
                DO $$ 
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns 
                        WHERE table_name='session_metadata' AND column_name='last_accessed'
                    ) THEN
                        ALTER TABLE session_metadata 
                        ADD COLUMN last_accessed TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP;
                    END IF;
                END $$;
            """)
            
            # Adiciona a coluna user_id se não existir
            cur.execute("""
                DO $$ 
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns 
                        WHERE table_name='session_metadata' AND column_name='user_id'
                    ) THEN
                        ALTER TABLE session_metadata 
                        ADD COLUMN user_id TEXT NOT NULL DEFAULT 'legacy';
                    END IF;
                END $$;
            """)
            
            # Cria índice para melhorar performance de queries por user_id
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_session_metadata_user_id 
                ON session_metadata(user_id);
            """)
            
            conn.commit()
            logger.info("✅ Tabela session_metadata configurada corretamente")
        
        return conn
    except Exception as e:
        st.error(f"❌ Erro ao conectar com o PostgreSQL: {e}")
        st.stop()

@st.cache_resource
def get_llms():
    logger.info("Criando instâncias dos LLMs (executado apenas uma vez)...")
    try:
        patient_llm = ChatOpenAI(model=model, openai_api_key=api_key, temperature=0, max_retries=2, timeout=30)
        evaluator_llm = ChatOpenAI(model=model, openai_api_key=api_key, temperature=0, max_retries=2, timeout=60)
        return patient_llm, evaluator_llm
    except Exception as e:
        st.error(f"❌ Erro ao inicializar modelos: {e}")
        st.stop()

@st.cache_resource
def get_app_and_checkpointer(_conn, _patient_llm, _evaluator_llm):
    logger.info("Configurando checkpointer e compilando o grafo (executado apenas uma vez)...")
    
    checkpointer = PostgresSaver(conn=_conn)
    
    _conn.autocommit = True
    checkpointer.setup()
    _conn.autocommit = False
    
    def patient_node(state: AgentState) -> Dict:
        system_prompt = SystemMessage(content=state["patient_prompt"])
        response = _patient_llm.invoke([system_prompt] + filter_messages(state["messages"]))
        return {"messages": [response]}

    def evaluation_1_node(state: AgentState) -> Dict:
        transcript = create_transcript(filter_messages(state["messages"]))
        response = _evaluator_llm.invoke(EVALUATION_SESSION_1.format(transcript=transcript))
        return {"messages": [AIMessage(content=response.content, response_metadata={EVALUATION_METADATA_KEY: True})], "current_session": 2, "session_1_end_index": len(state["messages"]) + 1}

    def evaluation_2_node(state: AgentState) -> Dict:
        transcript = create_transcript(filter_messages(state["messages"][state["session_1_end_index"]:]))
        response = _evaluator_llm.invoke(EVALUATION_SESSION_2.format(transcript=transcript))
        return {"messages": [AIMessage(content=response.content, response_metadata={EVALUATION_METADATA_KEY: True})], "current_session": 3}

    workflow = StateGraph(AgentState)
    workflow.add_node("patient_node", patient_node)
    workflow.add_node("evaluation_1_node", evaluation_1_node)
    workflow.add_node("evaluation_2_node", evaluation_2_node)
    workflow.add_conditional_edges(START, route_entry_point, {"patient_node": "patient_node", "evaluate_session_1": "evaluation_1_node", "evaluate_session_2": "evaluation_2_node"})
    workflow.add_edge("patient_node", END)
    workflow.add_edge("evaluation_1_node", END)
    workflow.add_edge("evaluation_2_node", END)

    app = workflow.compile(checkpointer=checkpointer)
    logger.info("✅ Aplicação LangGraph compilada e pronta para uso.")
    return app, checkpointer

# --- 6. INICIALIZAÇÃO DA APLICAÇÃO ---
db_connection = get_db_connection()
patient_llm, evaluator_llm = get_llms()
app, checkpointer = get_app_and_checkpointer(db_connection, patient_llm, evaluator_llm)

# --- 6.5. GERENCIAMENTO DE USER_ID E VALIDAÇÃO DE WHITELIST ---

def is_user_authorized(user_id: str) -> bool:
    """Verifica se o user_id está na whitelist."""
    is_authorized = user_id in ALLOWED_USER_IDS
    if not is_authorized:
        logger.warning(f"🚫 Tentativa de acesso não autorizado: {user_id}")
    return is_authorized

def get_or_create_user_id():
    """Gera ou recupera um user_id único e valida contra a whitelist."""
    
    # Tenta recuperar da URL (para compartilhamento)
    url_user_id = st.query_params.get("user")
    
    # Se existe na URL e não está no session_state, usa
    if url_user_id and "user_id" not in st.session_state:
        st.session_state.user_id = url_user_id
        logger.info(f"User ID recuperado da URL: {url_user_id}")
    
    # Se não existe no session_state, cria novo
    if "user_id" not in st.session_state:
        new_user_id = secrets.token_urlsafe(16)
        st.session_state.user_id = new_user_id
        st.query_params.user = new_user_id
        logger.info(f"Novo User ID criado: {new_user_id}")
    
    # Se existe no session_state mas não na URL, atualiza URL
    elif st.query_params.get("user") != st.session_state.user_id:
        st.query_params.user = st.session_state.user_id
    
    return st.session_state.user_id

def show_unauthorized_page():
    """Exibe página de acesso negado para usuários não autorizados."""
    st.set_page_config(page_title="Acesso Negado", page_icon="🚫")
    
    st.error("# 🚫 Acesso Negado")
    st.markdown("""
    ### Você não tem permissão para acessar este aplicativo.
    
    Este é um sistema restrito para uso exclusivo de participantes autorizados do **Project Match**.
    
    #### Por que vejo esta mensagem?
    - Você não está na lista de usuários autorizados
    - Seu link de acesso pode ter expirado
    - Você pode estar usando um link de outra pessoa
    
    #### Como obter acesso?
    Se você acredita que deveria ter acesso a este sistema, entre em contato com o administrador do projeto.
    
    ---
    
    *Project Match - Simulador de Terapia*
    """)
    
    with st.expander("ℹ️ Informações Técnicas"):
        st.code(f"User ID: {st.session_state.get('user_id', 'N/A')}")
        st.caption("Compartilhe este ID com o administrador para solicitar acesso.")
    
    st.stop()

# --- 7. FUNÇÕES DE GERENCIAMENTO DE SESSÃO ---

def update_session_access_time(thread_id: str):
    """Atualiza o timestamp de último acesso da sessão."""
    try:
        with db_connection.cursor() as cur:
            cur.execute("""
                UPDATE session_metadata 
                SET last_accessed = CURRENT_TIMESTAMP 
                WHERE thread_id = %s
            """, (thread_id,))
            db_connection.commit()
    except Exception as e:
        logger.error(f"Erro ao atualizar tempo de acesso: {e}")
        db_connection.rollback()

def get_recent_sessions(limit: int = 50):
    """Retorna as sessões mais recentes DO USUÁRIO ATUAL, ordenadas pela data de criação."""
    try:
        user_id = st.session_state.user_id
        
        with db_connection.cursor() as cur:
            cur.execute("""
                SELECT thread_id, persona_name, created_at, last_accessed 
                FROM session_metadata 
                WHERE user_id = %s
                ORDER BY created_at DESC 
                LIMIT %s
            """, (user_id, limit))
            results = cur.fetchall()
            db_connection.commit() 
            return results
    except Exception as e:
        logger.error(f"Erro ao buscar sessões recentes: {e}")
        db_connection.rollback() 
        return []

def save_session_metadata(thread_id: str, persona_name: str):
    """Salva os metadados da sessão no banco de dados COM user_id."""
    try:
        user_id = st.session_state.user_id
        
        with db_connection.cursor() as cur:
            cur.execute("""
                INSERT INTO session_metadata (thread_id, persona_name, user_id, last_accessed)
                VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
                ON CONFLICT (thread_id) DO UPDATE 
                SET persona_name = EXCLUDED.persona_name,
                    last_accessed = CURRENT_TIMESTAMP,
                    user_id = EXCLUDED.user_id
            """, (thread_id, persona_name, user_id))
            db_connection.commit()
        logger.info(f"Metadados salvos: {thread_id} -> {persona_name} (user: {user_id})")
    except Exception as e:
        logger.error(f"Erro ao salvar metadados: {e}")
        db_connection.rollback()

def load_session_metadata(thread_id: str) -> str:
    """Carrega o nome da persona do banco de dados SE pertencer ao usuário atual."""
    try:
        user_id = st.session_state.user_id
        
        with db_connection.cursor() as cur:
            cur.execute("""
                SELECT persona_name FROM session_metadata 
                WHERE thread_id = %s AND user_id = %s
            """, (thread_id, user_id))
            result = cur.fetchone()
            db_connection.commit()
            
            if result:
                update_session_access_time(thread_id)
                return result[0]
            else:
                logger.warning(f"Tentativa de acessar sessão de outro usuário: {thread_id}")
    except Exception as e:
        logger.error(f"Erro ao carregar metadados: {e}")
        db_connection.rollback()
    return None

def load_session_from_checkpoint(thread_id: str) -> bool:
    """Tenta carregar uma sessão existente do checkpoint. Retorna True se bem-sucedido."""
    try:
        logger.info(f"Tentando carregar sessão existente: {thread_id}")
        
        # Primeiro, tenta carregar os metadados da nossa tabela
        persona_name = load_session_metadata(thread_id)
        
        if not persona_name:
            logger.info("Metadados da sessão não encontrados ou sessão pertence a outro usuário")
            return False
        
        persona_data = next((p for p in PERSONAS_DATA if p["name"] == persona_name), None)
        if not persona_data:
            logger.warning(f"Persona {persona_name} não encontrada nos dados")
            return False
        
        # Agora tenta carregar o estado do checkpoint
        config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
        saved_state = checkpointer.get(config)
        
        # Inicializa com valores padrão
        messages = []
        current_session = 1
        session_1_end_index = 0
        
        # Se o checkpoint existir e tiver dados, usa eles
        if saved_state and saved_state.get("channel_values"):
            channel_values = saved_state["channel_values"]
            messages = channel_values.get("messages", [])
            current_session = channel_values.get("current_session", 1)
            session_1_end_index = channel_values.get("session_1_end_index", 0)
        
        # Restaura o estado no Streamlit
        st.session_state.messages = messages
        st.session_state.current_session_num = current_session
        st.session_state.session_1_end_index = session_1_end_index
        st.session_state.session_2_end_index = 0
        st.session_state.thread_id = thread_id
        st.session_state.current_patient = persona_data
        
        logger.info(f"Sessão {thread_id} restaurada com a persona {persona_name} ({len(messages)} mensagens)")
        if messages:
            st.toast(f"Sessão restaurada com {persona_name}!")
        return True
                    
    except Exception as e:
        logger.warning(f"Erro ao carregar sessão: {e}")
        import traceback
        logger.warning(traceback.format_exc())
    
    return False

def initialize_session(thread_id: str = None, force_new: bool = False):
    """Inicializa a sessão. Se um thread_id for fornecido, carrega o estado. Senão, cria uma nova sessão."""
    
    # Validar thread_id se fornecido
    if thread_id and not is_valid_uuid(thread_id):
        logger.warning(f"Thread ID inválido fornecido: {thread_id}")
        st.warning("⚠️ Link de sessão inválido. Criando nova sessão...")
        thread_id = None
    
    # Se for forçar nova sessão, pula a tentativa de carregar
    if thread_id and not force_new:
        if load_session_from_checkpoint(thread_id):
            return
    
    # Criar nova sessão - seleciona próximo paciente na ordem
    logger.info("Criando uma nova sessão.")
    new_thread_id = str(uuid.uuid4())
    
    # Busca a última sessão do usuário para determinar o próximo paciente
    recent_sessions = get_recent_sessions(limit=1)
    
    if recent_sessions and len(recent_sessions) > 0:
        last_persona_name = recent_sessions[0][1]  # Nome da persona da última sessão
        new_patient = get_next_persona(last_persona_name)
        logger.info(f"Última sessão foi com {last_persona_name}, próximo será {new_patient['name']}")
    else:
        # Se for a primeira sessão, começa com Clara
        new_patient = PERSONAS_DATA[0]
        logger.info(f"Primeira sessão do usuário, começando com {new_patient['name']}")
    
    # Salvar metadados da sessão
    save_session_metadata(new_thread_id, new_patient['name'])
    
    # Inicializar o estado da sessão no Streamlit
    st.session_state.messages = []
    st.session_state.thread_id = new_thread_id
    st.session_state.current_patient = new_patient
    st.session_state.current_session_num = 1
    st.session_state.session_1_end_index = 0
    st.session_state.session_2_end_index = 0
    
    # Atualiza a query string
    st.query_params.thread_id = new_thread_id
    logger.info(f"✅ Nova sessão inicializada: {new_thread_id} com a persona {new_patient['name']}")
    st.toast(f"✅ Novo paciente: {new_patient['name']}!")

# --- 8. VALIDAÇÃO DE ACESSO E INICIALIZAÇÃO ---

# Validar whitelist ANTES de qualquer operação
get_or_create_user_id()

if not is_user_authorized(st.session_state.user_id):
    show_unauthorized_page()

# Se chegou aqui, usuário está autorizado - continuar normalmente
logger.info(f"✅ Usuário autorizado: {st.session_state.user_id}")

# --- Lógica de Inicialização da Sessão ---
# Verifica se há um thread_id diferente na URL
url_thread_id = st.query_params.get("thread_id")
current_thread_id = st.session_state.get("thread_id")

if url_thread_id and url_thread_id != current_thread_id:
    # URL mudou - carregar nova sessão
    logger.info(f"Detectada mudança de thread_id: {current_thread_id} -> {url_thread_id}")
    initialize_session(url_thread_id)
elif "thread_id" not in st.session_state:
    # Primeira inicialização
    initialize_session(url_thread_id)

st.title("Simulador de Terapia (Project Match)")

# --- Barra Lateral (Sidebar) ---
with st.sidebar:
    st.title("Painel de Controle")
    components.html(CLOCK_HTML, height=65)
    
    st.header("Status da Simulação")
    if st.session_state.current_session_num <= 2:
        st.info(f"Sessão: **{st.session_state.current_session_num}** | Paciente: **{st.session_state.current_patient['name']}**", icon="⚠️")
    else:
        st.success("Simulação Concluída!", icon="✅")
    
    with st.expander("ℹ️ Informações de Acesso", expanded=False):
        st.caption(f"✅ Acesso Autorizado")
        st.caption(f"Thread ID: {st.session_state.thread_id}")
        st.caption(f"📅 Sessão iniciada: {datetime.now(BRAZIL_TZ).strftime('%H:%M')}")
        
    st.header("Suas Conversas")
    recent_sessions = get_recent_sessions(limit=50)

    if recent_sessions:
        current_tid = st.session_state.get("thread_id")
        
        for thread_id, persona, created_at, last_accessed in recent_sessions:
            is_current = thread_id == current_tid
            
            # Calcula o tempo desde último acesso
            now_utc = datetime.now(timezone.utc)
            last_accessed_utc = last_accessed if last_accessed.tzinfo else last_accessed.replace(tzinfo=timezone.utc)
            time_diff = now_utc - last_accessed_utc
            
            if time_diff.days > 0:
                time_str = f"Último acesso {time_diff.days}d atrás"
            elif time_diff.seconds > 3600:
                time_str = f"Último acesso {time_diff.seconds // 3600}h atrás"
            else:
                time_str = f"Último acesso {time_diff.seconds // 60}min atrás"
            
            # Formata a data de criação
            created_at_local = created_at.astimezone(BRAZIL_TZ)
            date_str = created_at_local.strftime('%d/%m %H:%M')
            
            button_label = f"{'🟢' if is_current else '⚪'} {persona} - {date_str}"

            if st.button(
                button_label,
                key=f"session_{thread_id}",
                use_container_width=True,
                disabled=is_current,
                help=time_str
            ):
                # Limpa o estado atual antes de carregar nova sessão
                for key in ['messages', 'current_session_num', 'session_1_end_index', 
                        'session_2_end_index', 'thread_id', 'current_patient']:
                    if key in st.session_state:
                        del st.session_state[key]
                
                # Atualiza URL e força carregamento
                st.query_params.thread_id = thread_id
                st.rerun()
    else:
        st.caption("Nenhuma conversa anterior")
        st.caption("Inicie uma nova simulação!")
    
    st.header("Controles")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Novo Paciente", use_container_width=True):
            initialize_session(force_new=True)
            st.rerun()
    with col2:
        def export_session_history():
            now_local = datetime.now(BRAZIL_TZ)
            output = [f"Paciente: {st.session_state.current_patient['name']}\nData: {now_local.strftime('%d/%m/%Y %H:%M:%S')}\n\n"]
            for msg in st.session_state.messages:
                if isinstance(msg, AIMessage) and msg.response_metadata.get(EVALUATION_METADATA_KEY):
                    output.append(f"\n--- AVALIAÇÃO ---\n{msg.content}\n-----------------\n")
                elif isinstance(msg, AIMessage):
                    output.append(f"Paciente: {msg.content}\n")
                elif isinstance(msg, HumanMessage) and END_SESSION_CODE not in msg.content:
                    output.append(f"Terapeuta: {msg.content}\n")
            return "".join(output)
        if st.session_state.messages:
            st.download_button("💾 Download", export_session_history(), f"sessao_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", "text/plain", use_container_width=True)
        else:
            st.button("💾 Download", use_container_width=True, disabled=True)
    
    if st.session_state.current_session_num <= 2:
        if st.button("🏁 Encerrar Sessão e Avaliar", type="primary", use_container_width=True):
            with st.spinner("⏳ Gerando avaliação detalhada..."):
                try:
                    response = app.invoke(
                        {
                            "messages": st.session_state.messages + [HumanMessage(content=END_SESSION_CODE)], 
                            "current_session": st.session_state.current_session_num, 
                            "session_1_end_index": st.session_state.session_1_end_index, 
                            "patient_prompt": st.session_state.current_patient['prompt'],
                            "persona_name": st.session_state.current_patient['name']
                        }, 
                        {"configurable": {"thread_id": st.session_state.thread_id}}
                    )
                    st.session_state.messages.append(response["messages"][-1])
                    if "current_session" in response:
                        new_session_num = response["current_session"]
                        if new_session_num == 2:
                            st.session_state.session_1_end_index = response.get("session_1_end_index", len(st.session_state.messages))
                            st.session_state.current_session_num = new_session_num
                            st.toast("✅ Sessão 1 avaliada! Iniciando Sessão 2...")
                        else:
                            st.session_state.session_2_end_index = len(st.session_state.messages)
                            st.session_state.current_session_num = new_session_num
                            st.toast("✅ Simulação concluída!")
                    st.rerun()
                except Exception as e:
                    logger.error(f"Error during evaluation: {e}")
                    st.error(f"❌ Erro crítico ao processar a avaliação: {str(e)}")

# --- Lógica de renderização do Chat ---
for i, msg in enumerate(st.session_state.messages):
    if isinstance(msg, AIMessage) and msg.response_metadata.get(EVALUATION_METADATA_KEY):
        with st.chat_message("assistant", avatar="📋"): st.markdown("### 📊 Avaliação da Sessão\n" + msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant", avatar="🧑‍⚕️").write(msg.content)
    elif isinstance(msg, HumanMessage) and END_SESSION_CODE not in msg.content:
        st.chat_message("user", avatar="👨‍💻").write(msg.content)

    if st.session_state.session_1_end_index > 0 and i == st.session_state.session_1_end_index - 1:
        st.divider(); st.subheader("🔄 Sessão 2"); st.divider()
    if st.session_state.session_2_end_index > 0 and i == st.session_state.session_2_end_index - 1:
        st.divider(); st.subheader("✅ Fim da Simulação"); st.divider()
        st.info("💡 Use o botão 'Download' para salvar o histórico ou 'Novo Paciente' para continuar.", icon="ℹ️")

# --- Input do Chat e Geração de Resposta ---
if prompt := st.chat_input("Digite sua mensagem...", disabled=(st.session_state.current_session_num > 2)):
    if not prompt.strip():
        st.warning("⚠️ Por favor, digite uma mensagem válida.")
    else:
        st.session_state.messages.append(HumanMessage(content=prompt))
        st.rerun()

if st.session_state.messages and isinstance(st.session_state.messages[-1], HumanMessage) and END_SESSION_CODE not in st.session_state.messages[-1].content:
    with st.chat_message("assistant", avatar="🧑‍⚕️"):
        with st.spinner("💭 Paciente Digitando..."):
            try:
                response = app.invoke(
                    {
                        "messages": st.session_state.messages, 
                        "current_session": st.session_state.current_session_num, 
                        "session_1_end_index": st.session_state.session_1_end_index, 
                        "patient_prompt": st.session_state.current_patient['prompt'],
                        "persona_name": st.session_state.current_patient['name']
                    },
                    {"configurable": {"thread_id": st.session_state.thread_id}}
                )
                ai_response = response["messages"][-1]
                st.session_state.messages.append(ai_response)
                st.rerun()
            except Exception as e:
                logger.error(f"Error generating patient response: {e}")
                st.error(f"❌ Erro ao gerar resposta: {str(e)}")
