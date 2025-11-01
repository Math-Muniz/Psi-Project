# Psi-Project - Simulador de Terapia (Project Match)

## Descrição do Projeto

O **Psi-Project** é um simulador de terapia cognitivo-comportamental (TCC) desenvolvido para treinamento e avaliação de terapeutas no tratamento de dependência de álcool. O sistema implementa o protocolo do **Project MATCH** (Matching Alcoholism Treatments to Client Heterogeneity), um dos estudos mais importantes sobre tratamento de alcoolismo.

O simulador oferece uma experiência realista de atendimento terapêutico através de pacientes virtuais com personalidades, histórias e resistências únicas, permitindo que terapeutas pratiquem suas habilidades em um ambiente controlado e recebam avaliações detalhadas sobre sua aderência ao protocolo.

## Funcionalidades Principais

### 1. Pacientes Virtuais (Personas)

O sistema inclui três personas de pacientes com diferentes perfis e desafios:

- **Clara (35 anos)** - Mãe em tempo integral, casada, consumo de vinho escondido
  - AUDIT Score: 19
  - Estado: Culpada, solitária, choro fácil
  - Medo central: Perder os filhos ou ser vista como mãe incapaz

- **Rafael (22 anos)** - Estudante de engenharia, bebedor social intenso
  - AUDIT Score: 22
  - Estado: Defensivo, sarcástico, resistente
  - Medo central: Isolamento social, perder amigos

- **Luiz (41 anos)** - Mestre de obras, bebe com colegas após trabalho
  - AUDIT Score: 14
  - Estado: Pressionado, irritado, minimiza o problema
  - Medo central: Perder respeito dos colegas de trabalho

### 2. Protocolo de 7 Sessões

O sistema implementa as 7 sessões do protocolo de Coping Skills Training:

1. **Sessão 1**: Introduction to Coping Skills Training
2. **Sessão 2**: Coping With Cravings and Urges to Drink
3. **Sessão 3**: Managing Thoughts About Alcohol and Drinking
4. **Sessão 4**: Problem Solving
5. **Sessão 5**: Drink Refusal Skills
6. **Sessão 6**: Planning for Emergencies and Coping With a Lapse
7. **Sessão 7**: Seemingly Irrelevant Decisions

### 3. Sistema de Avaliação

Após cada sessão, o sistema gera uma avaliação detalhada que verifica:
- Checklist de componentes obrigatórios do protocolo
- Análise ponto a ponto da aderência
- Pontos fortes e áreas de melhoria
- Recomendações construtivas

### 4. Gerenciamento de Sessões

- Histórico completo de conversas
- Persistência de dados em PostgreSQL (Supabase)
- Sistema de autenticação por whitelist
- Exportação de transcrições em formato texto

## Tecnologias Utilizadas

### Backend
- **Python 3.x**
- **LangChain** - Framework para aplicações com LLMs
- **LangGraph** - Orquestração de fluxos conversacionais
- **OpenAI GPT-4** - Modelo de linguagem para pacientes e avaliador

### Frontend
- **Streamlit** - Interface web interativa

### Banco de Dados
- **PostgreSQL (Supabase)** - Persistência de conversas e checkpoints
- **psycopg** - Driver PostgreSQL

### Gerenciamento
- **python-dotenv** - Gerenciamento de variáveis de ambiente

## Configuração do Ambiente

### 1. Pré-requisitos

- Python 3.8+
- Conta OpenAI com API key
- Conta Supabase (PostgreSQL)

### 2. Instalação

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/Psi-Project.git
cd Psi-Project

# Instale as dependências
pip install -r requirements.txt
```

### 3. Configuração das Variáveis de Ambiente

Crie um arquivo `.env` na raiz do projeto com as seguintes variáveis:

```env
# OpenAI
OPENAI_API_KEY=sua_chave_aqui
OPENAI_MODEL=gpt-4o-mini

# PostgreSQL/Supabase
POSTGRES_USER=seu_usuario
POSTGRES_PASSWORD=sua_senha
POSTGRES_HOST=seu_host.supabase.co
POSTGRES_PORT=6543
POSTGRES_DB=postgres

# Whitelist de Usuários (separados por vírgula)
ALLOWED_USER_IDS=user1,user2,user3
```

### 4. Executar a Aplicação

```bash
streamlit run app.py
```

A aplicação estará disponível em `http://localhost:8501`

## Estrutura do Projeto

```
Psi-Project/
├── app.py              # Aplicação principal Streamlit
├── prompts.py          # Definições de personas e prompts de avaliação
├── README.md           # Documentação do projeto
├── .env               # Variáveis de ambiente (não versionado)
└── requirements.txt   # Dependências Python
```

## Como Usar

### Para Terapeutas em Treinamento

1. **Acesso**: Entre com seu user_id autorizado na URL
2. **Início de Sessão**: O sistema atribui automaticamente um paciente (rotação entre Clara, Rafael e Luiz)
3. **Condução da Terapia**: Conduza a sessão seguindo o protocolo da TCC
4. **Encerramento**: Clique em "Encerrar Sessão e Avaliar"
5. **Feedback**: Receba avaliação detalhada da sua aderência ao protocolo
6. **Próxima Sessão**: Continue com as sessões subsequentes (2-7)
7. **Exportação**: Baixe a transcrição para revisão posterior

### Recursos da Interface

- **Painel de Controle**: Status da sessão atual, progresso
- **Histórico**: Acesso a conversas anteriores
- **Relógio Digital**: Controle de tempo em tempo real
- **Download**: Exportação de transcrições
- **Novo Paciente**: Iniciar nova simulação com outro paciente

## Segurança e Autenticação

O sistema implementa:
- Whitelist de usuários autorizados via `ALLOWED_USER_IDS`
- Isolamento de dados por user_id
- Tokens de sessão únicos (UUID)
- Conexão segura com banco de dados

## Dados Clínicos

### AUDIT Scores
O sistema simula pacientes com diferentes níveis de gravidade segundo o AUDIT (Alcohol Use Disorders Identification Test):

- **8-15 pontos**: Uso de risco (Luiz - 14)
- **16-19 pontos**: Uso nocivo (Clara - 19)
- **20-40 pontos**: Possível dependência (Rafael - 22)

## Referências

- **Project MATCH Research Group** - Manual de Terapia Cognitivo-Comportamental
- **AUDIT** - Alcohol Use Disorders Identification Test (OMS)

## Licença

Este projeto é desenvolvido para fins educacionais e de treinamento profissional em psicologia clínica.
