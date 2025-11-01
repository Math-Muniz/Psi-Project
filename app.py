import os
import streamlit as st
import streamlit.components.v1 as components
import uuid
import logging
import secrets
from typing import List, Annotated, TypedDict, Dict, Optional
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tracers.context import tracing_v2_enabled
from langgraph.graph import StateGraph, add_messages, END, START
from langgraph.checkpoint.postgres import PostgresSaver
import psycopg
from psycopg.rows import dict_row
from prompts import (
    PERSONA_RAFAEL, 
    PERSONA_CLARA, 
    PERSONA_LUIZ, 
    EVALUATION_SESSION_1, 
    EVALUATION_SESSION_2,
    EVALUATION_SESSION_3,
    EVALUATION_SESSION_4,
    EVALUATION_SESSION_5,
    EVALUATION_SESSION_6,
    EVALUATION_SESSION_7
)

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

# Mapeia número de sessão para prompt de avaliação
EVALUATION_PROMPTS = {
    1: EVALUATION_SESSION_1,
    2: EVALUATION_SESSION_2,
    3: EVALUATION_SESSION_3,
    4: EVALUATION_SESSION_4,
    5: EVALUATION_SESSION_5,
    6: EVALUATION_SESSION_6,
    7: EVALUATION_SESSION_7
}

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    current_session: int
    session_end_indices: Dict[int, int]
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
    """Remove mensagens de controle e avaliações."""
    return [msg for msg in messages if not (isinstance(msg, HumanMessage) and END_SESSION_CODE in msg.content) and not (isinstance(msg, AIMessage) and msg.response_metadata.get(EVALUATION_METADATA_KEY))]

def create_transcript(messages: List[BaseMessage]) -> str:
    """Cria transcrição formatada das mensagens."""
    return "\n".join([f"{'Terapeuta' if isinstance(msg, HumanMessage) else 'Paciente'}: {msg.content}" for msg in messages])

def get_session_messages(state: AgentState, session_number: int) -> List[BaseMessage]:
    """Extrai apenas as mensagens de uma sessão específica."""
    session_end_indices = state.get("session_end_indices", {})
    all_messages = state["messages"]
    
    if session_number == 1:
        start_idx = 0
    else:
        start_idx = session_end_indices.get(session_number - 1, 0)
    
    end_idx = len(all_messages)
    
    return all_messages[start_idx:end_idx]

def route_entry_point(state: AgentState) -> str:
    """Roteia para o nó correto baseado no estado."""
    last_message = state["messages"][-1] if state["messages"] else None
    
    if isinstance(last_message, HumanMessage) and END_SESSION_CODE in last_message.content:
        current_session = state.get("current_session", 1)
        
        if current_session == 1:
            return "evaluate_session_1"
        elif current_session == 2:
            return "evaluate_session_2"
        elif current_session == 3:
            return "evaluate_session_3"
        elif current_session == 4:
            return "evaluate_session_4"
        elif current_session == 5:
            return "evaluate_session_5"
        elif current_session == 6:
            return "evaluate_session_6"
        elif current_session == 7:
            return "evaluate_session_7"
    
    return "patient_node"

def get_next_persona(current_persona_name: str) -> Dict:
    """Retorna o próximo paciente na ordem fixa: Clara → Rafael → Luiz → Clara..."""
    current_persona = next((p for p in PERSONAS_DATA if p["name"] == current_persona_name), None)
    
    if not current_persona:
        return PERSONAS_DATA[0]
    
    current_order = current_persona["order"]
    next_persona = next((p for p in PERSONAS_DATA if p["order"] == current_order + 1), None)
    
    if not next_persona:
        next_persona = PERSONAS_DATA[0]
    
    return next_persona

# --- 5. POOL DE CONEXÕES SUPABASE ---

class SupabaseConnectionPool:
    """Pool de conexões otimizado para Supabase."""
    
    def __init__(self, min_size: int = 2, max_size: int = 10):
        self.connection_string = self._build_connection_string()
        self.min_size = min_size
        self.max_size = max_size
        self._pool = None
        
    def _build_connection_string(self) -> str:
        """Constrói string de conexão otimizada para Supabase."""
        return (
            f"host={os.getenv('POSTGRES_HOST')} "
            f"port={os.getenv('POSTGRES_PORT')} "
            f"dbname={os.getenv('POSTGRES_DB')} "
            f"user={os.getenv('POSTGRES_USER')} "
            f"password={os.getenv('POSTGRES_PASSWORD')} "
            f"sslmode=require "
            f"connect_timeout=10 "
            f"keepalives=1 "
            f"keepalives_idle=30 "
            f"keepalives_interval=10 "
            f"keepalives_count=5 "
            f"application_name=project-match-simulator"
        )
    
    def get_connection(self):
        """Obtém conexão do pool."""
        try:
            if self._pool is None:
                logger.info("Criando pool de conexões Supabase...")
                from psycopg_pool import ConnectionPool
                self._pool = ConnectionPool(
                    self.connection_string,
                    min_size=self.min_size,
                    max_size=self.max_size,
                    timeout=30,
                    max_waiting=20
                )
                logger.info(f"✅ Pool criado: {self.min_size}-{self.max_size} conexões")
            
            return self._pool.connection()
        except ImportError:
            # Fallback se psycopg_pool não estiver disponível
            logger.warning("psycopg_pool não disponível, usando conexão simples")
            return psycopg.connect(self.connection_string)
        except Exception as e:
            logger.error(f"Erro ao obter conexão do pool: {e}")
            # Fallback para conexão direta
            return psycopg.connect(self.connection_string)

# --- 6. FUNÇÕES DE BANCO DE DADOS OTIMIZADAS ---

@st.cache_resource
def get_connection_pool():
    """Inicializa pool de conexões (cached)."""
    return SupabaseConnectionPool(min_size=2, max_size=10)

def execute_db_query(query: str, params: tuple = None, fetch: bool = False):
    """
    Executa query no Supabase com retry automático e logging.
    
    Args:
        query: SQL query
        params: Parâmetros da query
        fetch: Se True, retorna resultados (SELECT)
    """
    pool = get_connection_pool()
    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            with pool.get_connection() as conn:
                with conn.cursor(row_factory=dict_row) as cur:
                    cur.execute(query, params or ())
                    
                    if fetch:
                        results = cur.fetchall()
                        conn.commit()
                        return results
                    else:
                        conn.commit()
                        return None
        
        except Exception as e:
            logger.warning(f"Tentativa {attempt + 1}/{max_retries} falhou: {e}")
            
            if attempt < max_retries - 1:
                import time
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                logger.error(f"Erro após {max_retries} tentativas: {e}")
                raise

def setup_database():
    """Configura tabelas e índices otimizados para Supabase."""
    logger.info("Configurando tabelas no Supabase...")
    
    queries = [
        # Tabela principal com índices otimizados
        """
        CREATE TABLE IF NOT EXISTS session_metadata (
            thread_id TEXT PRIMARY KEY,
            persona_name TEXT NOT NULL,
            user_id TEXT NOT NULL DEFAULT 'legacy',
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            last_accessed TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            session_count INTEGER DEFAULT 0,
            total_messages INTEGER DEFAULT 0
        )
        """,
        
        # Índice composto para queries do usuário
        """
        CREATE INDEX IF NOT EXISTS idx_user_sessions 
        ON session_metadata(user_id, created_at DESC)
        """,
        
        # Índice para cleanup de sessões antigas
        """
        CREATE INDEX IF NOT EXISTS idx_last_accessed 
        ON session_metadata(last_accessed)
        WHERE last_accessed < CURRENT_TIMESTAMP - INTERVAL '90 days'
        """,
        
        # Tabela de métricas (novo)
        """
        CREATE TABLE IF NOT EXISTS session_metrics (
            id SERIAL PRIMARY KEY,
            thread_id TEXT REFERENCES session_metadata(thread_id) ON DELETE CASCADE,
            session_number INTEGER NOT NULL,
            evaluation_score JSONB,
            message_count INTEGER,
            duration_seconds INTEGER,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        )
        """,
        
        # Índice para métricas
        """
        CREATE INDEX IF NOT EXISTS idx_metrics_thread 
        ON session_metrics(thread_id, session_number)
        """
    ]
    
    for query in queries:
        try:
            execute_db_query(query)
        except Exception as e:
            logger.error(f"Erro ao executar query de setup: {e}")
    
    logger.info("✅ Database configurado com sucesso")

# --- 7. FUNÇÕES CACHEADAS PARA RECURSOS CAROS ---

@st.cache_resource
def get_llms():
    """Inicializa LLMs com configuração otimizada."""
    logger.info("Criando instâncias dos LLMs...")
    try:
        # LLM do paciente com streaming
        patient_llm = ChatOpenAI(
            model=model,
            openai_api_key=api_key,
            temperature=0,
            max_retries=3,
            timeout=30,
            streaming=False,  # Desabilitado para Streamlit
            model_kwargs={
                "top_p": 0.95,
                "frequency_penalty": 0.3,
                "presence_penalty": 0.3
            }
        )
        
        # LLM avaliador com timeout maior
        evaluator_llm = ChatOpenAI(
            model=model,
            openai_api_key=api_key,
            temperature=0,
            max_retries=3,
            timeout=90,
            model_kwargs={
                "top_p": 1.0,
                "frequency_penalty": 0,
                "presence_penalty": 0
            }
        )
        
        logger.info("✅ LLMs inicializados com sucesso")
        return patient_llm, evaluator_llm
    except Exception as e:
        st.error(f"❌ Erro ao inicializar modelos: {e}")
        st.stop()

@st.cache_resource
def get_app_and_checkpointer(_patient_llm, _evaluator_llm):
    """Compila grafo LangGraph com observabilidade LangSmith."""
    logger.info("Compilando grafo LangGraph...")
    
    # Pool de conexões para checkpointer
    pool = get_connection_pool()
    conn = pool.get_connection()
    
    checkpointer = PostgresSaver(conn=conn)
    
    conn.autocommit = True
    checkpointer.setup()
    conn.autocommit = False
    
    def patient_node(state: AgentState) -> Dict:
        """Nó que gera resposta do paciente com tracking LangSmith."""
        try:
            # Metadata para LangSmith
            metadata = {
                "session_number": state.get("current_session", 1),
                "persona": state.get("persona_name", "unknown"),
                "message_count": len(state.get("messages", [])),
                "node_type": "patient_response"
            }
            
            # Tags para filtrar no LangSmith
            tags = [
                f"session-{state.get('current_session', 1)}",
                f"persona-{state.get('persona_name', 'unknown')}",
                "patient-node"
            ]
            
            with tracing_v2_enabled(
                project_name=LANGSMITH_PROJECT,
                tags=tags,
                metadata=metadata
            ):
                system_prompt = SystemMessage(content=state["patient_prompt"])
                response = _patient_llm.invoke([system_prompt] + filter_messages(state["messages"]))
            
            return {"messages": [response]}
        
        except Exception as e:
            logger.error(f"Erro no patient_node: {e}")
            raise
    
    def create_evaluation_node(session_number: int):
        """Cria nó de avaliação com tracking detalhado."""
        def evaluation_node(state: AgentState) -> Dict:
            try:
                # Metadata rica para LangSmith
                session_messages = get_session_messages(state, session_number)
                filtered = filter_messages(session_messages)
                
                metadata = {
                    "session_number": session_number,
                    "persona": state.get("persona_name", "unknown"),
                    "messages_evaluated": len(filtered),
                    "total_messages": len(state.get("messages", [])),
                    "node_type": "evaluation",
                    "evaluation_type": f"session_{session_number}"
                }
                
                tags = [
                    f"evaluation-session-{session_number}",
                    f"persona-{state.get('persona_name', 'unknown')}",
                    "evaluation-node"
                ]
                
                with tracing_v2_enabled(
                    project_name=LANGSMITH_PROJECT,
                    tags=tags,
                    metadata=metadata
                ):
                    transcript = create_transcript(filtered)
                    evaluation_prompt = EVALUATION_PROMPTS[session_number]
                    response = _evaluator_llm.invoke(evaluation_prompt.format(transcript=transcript))
                
                # Salvar métricas
                try:
                    save_session_metrics(
                        state.get("thread_id", "unknown"),
                        session_number,
                        len(filtered)
                    )
                except Exception as e:
                    logger.warning(f"Erro ao salvar métricas: {e}")
                
                # Atualizar índices
                session_end_indices = state.get("session_end_indices", {}).copy()
                session_end_indices[session_number] = len(state["messages"]) + 1
                
                return {
                    "messages": [AIMessage(content=response.content, response_metadata={EVALUATION_METADATA_KEY: True})],
                    "current_session": session_number + 1,
                    "session_end_indices": session_end_indices
                }
            
            except Exception as e:
                logger.error(f"Erro no evaluation_node (sessão {session_number}): {e}")
                raise
        
        return evaluation_node
    
    # Criar workflow
    workflow = StateGraph(AgentState)
    workflow.add_node("patient_node", patient_node)
    
    for i in range(1, 8):
        workflow.add_node(f"evaluation_{i}_node", create_evaluation_node(i))
    
    workflow.add_conditional_edges(
        START,
        route_entry_point,
        {
            "patient_node": "patient_node",
            "evaluate_session_1": "evaluation_1_node",
            "evaluate_session_2": "evaluation_2_node",
            "evaluate_session_3": "evaluation_3_node",
            "evaluate_session_4": "evaluation_4_node",
            "evaluate_session_5": "evaluation_5_node",
            "evaluate_session_6": "evaluation_6_node",
            "evaluate_session_7": "evaluation_7_node"
        }
    )
    
    workflow.add_edge("patient_node", END)
    for i in range(1, 8):
        workflow.add_edge(f"evaluation_{i}_node", END)
    
    app = workflow.compile(checkpointer=checkpointer)
    logger.info("✅ Aplicação LangGraph compilada com 7 sessões e observabilidade LangSmith")
    return app, checkpointer

# --- 8. FUNÇÕES DE MÉTRICAS ---

def save_session_metrics(thread_id: str, session_number: int, message_count: int):
    """Salva métricas da sessão no Supabase."""
    query = """
        INSERT INTO session_metrics (thread_id, session_number, message_count)
        VALUES (%s, %s, %s)
        ON CONFLICT DO NOTHING
    """
    try:
        execute_db_query(query, (thread_id, session_number, message_count))
    except Exception as e:
        logger.warning(f"Erro ao salvar métricas: {e}")

def update_session_stats(thread_id: str, session_num: int, total_msgs: int):
    """Atualiza estatísticas da sessão."""
    query = """
        UPDATE session_metadata 
        SET session_count = %s, 
            total_messages = %s,
            last_accessed = CURRENT_TIMESTAMP
        WHERE thread_id = %s
    """
    try:
        execute_db_query(query, (session_num, total_msgs, thread_id))
    except Exception as e:
        logger.warning(f"Erro ao atualizar stats: {e}")

# --- 9. INICIALIZAÇÃO ---
setup_database()
patient_llm, evaluator_llm = get_llms()
app, checkpointer = get_app_and_checkpointer(patient_llm, evaluator_llm)

# --- 10. GERENCIAMENTO DE USER_ID ---

def is_user_authorized(user_id: str) -> bool:
    """Verifica se o user_id está na whitelist."""
    is_authorized = user_id in ALLOWED_USER_IDS
    if not is_authorized:
        logger.warning(f"🚫 Tentativa de acesso não autorizado: {user_id}")
    return is_authorized

def get_or_create_user_id():
    """Gera ou recupera um user_id único."""
    url_user_id = st.query_params.get("user")
    
    if url_user_id and "user_id" not in st.session_state:
        st.session_state.user_id = url_user_id
        logger.info(f"User ID recuperado da URL: {url_user_id}")
    
    if "user_id" not in st.session_state:
        new_user_id = secrets.token_urlsafe(16)
        st.session_state.user_id = new_user_id
        st.query_params.user = new_user_id
        logger.info(f"Novo User ID criado: {new_user_id}")
    
    elif st.query_params.get("user") != st.session_state.user_id:
        st.query_params.user = st.session_state.user_id
    
    return st.session_state.user_id

def show_unauthorized_page():
    """Exibe página de acesso negado."""
    st.error("# 🚫 Acesso Negado")
    st.markdown("""
    ### Você não tem permissão para acessar este aplicativo.
    
    Este é um sistema restrito para uso exclusivo de participantes autorizados do **Project Match**.
    
    #### Como obter acesso?
    Entre em contato com o administrador do projeto.
    """)
    
    with st.expander("ℹ️ Informações Técnicas"):
        st.code(f"User ID: {st.session_state.get('user_id', 'N/A')}")
    
    st.stop()

# --- 11. FUNÇÕES DE SESSÃO OTIMIZADAS ---

def get_recent_sessions(limit: int = 50) -> List[Dict]:
    """Retorna sessões recentes com dados completos."""
    query = """
        SELECT 
            thread_id, 
            persona_name, 
            created_at, 
            last_accessed,
            session_count,
            total_messages
        FROM session_metadata 
        WHERE user_id = %s
        ORDER BY created_at DESC 
        LIMIT %s
    """
    try:
        user_id = st.session_state.user_id
        results = execute_db_query(query, (user_id, limit), fetch=True)
        return results or []
    except Exception as e:
        logger.error(f"Erro ao buscar sessões: {e}")
        return []

def save_session_metadata(thread_id: str, persona_name: str):
    """Salva metadados da sessão."""
    query = """
        INSERT INTO session_metadata (thread_id, persona_name, user_id, last_accessed)
        VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
        ON CONFLICT (thread_id) DO UPDATE 
        SET persona_name = EXCLUDED.persona_name,
            last_accessed = CURRENT_TIMESTAMP,
            user_id = EXCLUDED.user_id
    """
    try:
        user_id = st.session_state.user_id
        execute_db_query(query, (thread_id, persona_name, user_id))
        logger.info(f"Metadados salvos: {thread_id} -> {persona_name}")
    except Exception as e:
        logger.error(f"Erro ao salvar metadados: {e}")

def load_session_metadata(thread_id: str) -> Optional[str]:
    """Carrega nome da persona."""
    query = """
        SELECT persona_name 
        FROM session_metadata 
        WHERE thread_id = %s AND user_id = %s
    """
    try:
        user_id = st.session_state.user_id
        results = execute_db_query(query, (thread_id, user_id), fetch=True)
        
        if results and len(results) > 0:
            # Atualizar last_accessed
            update_query = "UPDATE session_metadata SET last_accessed = CURRENT_TIMESTAMP WHERE thread_id = %s"
            execute_db_query(update_query, (thread_id,))
            return results[0]['persona_name']
    except Exception as e:
        logger.error(f"Erro ao carregar metadados: {e}")
    
    return None

def load_session_from_checkpoint(thread_id: str) -> bool:
    """Carrega sessão existente."""
    try:
        logger.info(f"Carregando sessão: {thread_id}")
        
        persona_name = load_session_metadata(thread_id)
        if not persona_name:
            return False
        
        persona_data = next((p for p in PERSONAS_DATA if p["name"] == persona_name), None)
        if not persona_data:
            return False
        
        config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
        saved_state = checkpointer.get(config)
        
        messages = []
        current_session = 1
        session_end_indices = {}
        
        if saved_state and saved_state.get("channel_values"):
            channel_values = saved_state["channel_values"]
            messages = channel_values.get("messages", [])
            current_session = channel_values.get("current_session", 1)
            session_end_indices = channel_values.get("session_end_indices", {})
        
        st.session_state.messages = messages
        st.session_state.current_session_num = current_session
        st.session_state.session_end_indices = session_end_indices
        st.session_state.thread_id = thread_id
        st.session_state.current_patient = persona_data
        
        logger.info(f"Sessão restaurada: {persona_name}, sessão {current_session}")
        if messages:
            st.toast(f"Sessão restaurada com {persona_name}!")
        return True
                    
    except Exception as e:
        logger.warning(f"Erro ao carregar sessão: {e}")
    
    return False

def initialize_session(thread_id: str = None, force_new: bool = False):
    """Inicializa sessão."""
    
    if thread_id and not is_valid_uuid(thread_id):
        logger.warning(f"Thread ID inválido: {thread_id}")
        st.warning("⚠️ Link inválido. Criando nova sessão...")
        thread_id = None
    
    if thread_id and not force_new:
        if load_session_from_checkpoint(thread_id):
            return
    
    logger.info("Criando nova sessão")
    new_thread_id = str(uuid.uuid4())
    
    recent_sessions = get_recent_sessions(limit=1)
    
    if recent_sessions and len(recent_sessions) > 0:
        last_persona_name = recent_sessions[0]['persona_name']
        new_patient = get_next_persona(last_persona_name)
        logger.info(f"Última: {last_persona_name}, próximo: {new_patient['name']}")
    else:
        new_patient = PERSONAS_DATA[0]
        logger.info(f"Primeira sessão: {new_patient['name']}")
    
    save_session_metadata(new_thread_id, new_patient['name'])
    
    st.session_state.messages = []
    st.session_state.thread_id = new_thread_id
    st.session_state.current_patient = new_patient
    st.session_state.current_session_num = 1
    st.session_state.session_end_indices = {}
    
    st.query_params.thread_id = new_thread_id
    logger.info(f"✅ Nova sessão: {new_thread_id}")
    st.toast(f"✅ Novo paciente: {new_patient['name']}!")

# --- 12. VALIDAÇÃO E INICIALIZAÇÃO ---

get_or_create_user_id()

if not is_user_authorized(st.session_state.user_id):
    show_unauthorized_page()

logger.info(f"✅ Usuário autorizado: {st.session_state.user_id}")

url_thread_id = st.query_params.get("thread_id")
current_thread_id = st.session_state.get("thread_id")

if url_thread_id and url_thread_id != current_thread_id:
    initialize_session(url_thread_id)
elif "thread_id" not in st.session_state:
    initialize_session(url_thread_id)

st.title("Simulador de Terapia (Project Match)")

# --- 13. INTERFACE (SIDEBAR E CHAT) ---
# [Código da interface permanece igual ao anterior, mas com get_recent_sessions retornando dicts]

with st.sidebar:
    st.title("Painel de Controle")
    components.html(CLOCK_HTML, height=65)
    
    st.header("Status da Simulação")
    if st.session_state.current_session_num <= 7:
        progress = (st.session_state.current_session_num - 1) / 7
        st.progress(progress)
        st.info(
            f"Sessão: **{st.session_state.current_session_num}/7** | "
            f"Paciente: **{st.session_state.current_patient['name']}**", 
            icon="⚠️"
        )
    else:
        st.success("✅ Todas as 7 sessões concluídas!", icon="🎉")
    
    # Link para LangSmith se disponível
    if LANGSMITH_API_KEY:
        with st.expander("🔍 Observabilidade LangSmith"):
            st.markdown(f"[Ver traces no LangSmith]({LANGSMITH_ENDPOINT}/o/default/projects/p/{LANGSMITH_PROJECT})")
            st.caption(f"Projeto: {LANGSMITH_PROJECT}")
    
    with st.expander("ℹ️ Informações de Acesso", expanded=False):
        st.caption(f"✅ Acesso Autorizado")
        st.caption(f"Thread ID: {st.session_state.thread_id}")
        st.caption(f"📅 Iniciada: {datetime.now(BRAZIL_TZ).strftime('%H:%M')}")
        
    st.header("Suas Conversas")
    recent_sessions = get_recent_sessions(limit=50)

    if recent_sessions:
        current_tid = st.session_state.get("thread_id")
        
        for session in recent_sessions:
            thread_id = session['thread_id']
            persona = session['persona_name']
            created_at = session['created_at']
            last_accessed = session['last_accessed']
            
            is_current = thread_id == current_tid
            
            now_utc = datetime.now(timezone.utc)
            last_accessed_utc = last_accessed if last_accessed.tzinfo else last_accessed.replace(tzinfo=timezone.utc)
            time_diff = now_utc - last_accessed_utc
            
            if time_diff.days > 0:
                time_str = f"{time_diff.days}d atrás"
            elif time_diff.seconds > 3600:
                time_str = f"{time_diff.seconds // 3600}h atrás"
            else:
                time_str = f"{time_diff.seconds // 60}min atrás"
            
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
                for key in ['messages', 'current_session_num', 'session_end_indices', 
                        'thread_id', 'current_patient']:
                    if key in st.session_state:
                        del st.session_state[key]
                
                st.query_params.thread_id = thread_id
                st.rerun()
    else:
        st.caption("Nenhuma conversa anterior")
    
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
            st.download_button(
                "💾 Download", 
                export_session_history(), 
                f"sessao_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", 
                "text/plain", 
                use_container_width=True
            )
        else:
            st.button("💾 Download", use_container_width=True, disabled=True)
    
    if st.session_state.current_session_num <= 7:
        if st.button("🏁 Encerrar Sessão e Avaliar", type="primary", use_container_width=True):
            with st.spinner("⏳ Gerando avaliação detalhada..."):
                try:
                    # Metadados para LangSmith
                    run_name = f"Evaluation-Session-{st.session_state.current_session_num}"
                    
                    response = app.invoke(
                        {
                            "messages": st.session_state.messages + [HumanMessage(content=END_SESSION_CODE)], 
                            "current_session": st.session_state.current_session_num, 
                            "session_end_indices": st.session_state.get("session_end_indices", {}),
                            "patient_prompt": st.session_state.current_patient['prompt'],
                            "persona_name": st.session_state.current_patient['name']
                        }, 
                        {"configurable": {"thread_id": st.session_state.thread_id}},
                        {"run_name": run_name}  # Nome da run no LangSmith
                    )
                    
                    st.session_state.messages.append(response["messages"][-1])
                    
                    if "current_session" in response:
                        new_session_num = response["current_session"]
                        st.session_state.current_session_num = new_session_num
                        
                        if "session_end_indices" in response:
                            st.session_state.session_end_indices = response["session_end_indices"]
                        
                        # Atualizar estatísticas
                        update_session_stats(
                            st.session_state.thread_id,
                            new_session_num - 1,
                            len(st.session_state.messages)
                        )
                        
                        if new_session_num <= 7:
                            st.toast(f"✅ Sessão {new_session_num - 1} avaliada! Iniciando Sessão {new_session_num}...")
                        else:
                            st.toast("🎉 Todas as 7 sessões concluídas!")
                    
                    st.rerun()
                except Exception as e:
                    logger.error(f"Erro durante avaliação: {e}")
                    st.error(f"❌ Erro ao processar avaliação: {str(e)}")

# --- Renderização e Input do Chat ---
session_end_indices = st.session_state.get("session_end_indices", {})

for i, msg in enumerate(st.session_state.messages):
    if isinstance(msg, AIMessage) and msg.response_metadata.get(EVALUATION_METADATA_KEY):
        with st.chat_message("assistant", avatar="📋"):
            st.markdown("### 📊 Avaliação da Sessão\n" + msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant", avatar="🧑‍⚕️").write(msg.content)
    elif isinstance(msg, HumanMessage) and END_SESSION_CODE not in msg.content:
        st.chat_message("user", avatar="👨‍💻").write(msg.content)
    
    for session_num in range(1, 8):
        if session_num in session_end_indices and i == session_end_indices[session_num] - 1:
            st.divider()
            if session_num < 7:
                st.subheader(f"🔄 Sessão {session_num + 1}")
            else:
                st.subheader("✅ Fim da Simulação")
                st.info("💡 Use 'Download' para salvar ou 'Novo Paciente' para continuar.", icon="ℹ️")
            st.divider()

if prompt := st.chat_input("Digite sua mensagem...", disabled=(st.session_state.current_session_num > 7)):
    if not prompt.strip():
        st.warning("⚠️ Por favor, digite uma mensagem válida.")
    else:
        st.session_state.messages.append(HumanMessage(content=prompt))
        st.rerun()

if st.session_state.messages and isinstance(st.session_state.messages[-1], HumanMessage) and END_SESSION_CODE not in st.session_state.messages[-1].content:
    with st.chat_message("assistant", avatar="🧑‍⚕️"):
        with st.spinner("💭 Paciente Digitando..."):
            try:
                # Nome da run para LangSmith
                run_name = f"Patient-Response-Session-{st.session_state.current_session_num}"
                
                response = app.invoke(
                    {
                        "messages": st.session_state.messages, 
                        "current_session": st.session_state.current_session_num, 
                        "session_end_indices": st.session_state.get("session_end_indices", {}),
                        "patient_prompt": st.session_state.current_patient['prompt'],
                        "persona_name": st.session_state.current_patient['name']
                    },
                    {"configurable": {"thread_id": st.session_state.thread_id}},
                    {"run_name": run_name}
                )
                ai_response = response["messages"][-1]
                st.session_state.messages.append(ai_response)
                st.rerun()
            except Exception as e:
                logger.error(f"Erro ao gerar resposta: {e}")
                st.error(f"❌ Erro ao gerar resposta: {str(e)}")
