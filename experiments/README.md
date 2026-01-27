# Bitcoin Agent Privacy Experiments

Sistema d'automatització d'experiments per avaluar la privacitat de les PSBTs generades per l'Agent Bitcoin amb diferents LLMs i configuracions.

## Estructura

```
experiments/
├── experiments.csv          # Definició dels experiments (editable amb Excel)
├── experiment_runner.py     # Script d'execució
├── results/                 # Resultats (generat automàticament)
│   ├── experiments_*.csv    # Resultats resumits
│   ├── experiments_*.json   # Resultats detallats
│   └── psbts/              # PSBTs generades
└── README.md               # Aquesta documentació
```

## Ús Ràpid

```bash
# Des del directori del projecte
cd experiments

# IMPORTANT: Usar el venv del projecte
source ../.venv/bin/activate

# Executar tots els experiments
cd /home/jaume/feina/bitcoin-agent-project/experiments && /home/jaume/feina/bitcoin-agent-project/.venv/bin/python experiment_runner.py experiments.csv
# Executar només experiments amb un tag específic
python experiment_runner.py experiments.csv --filter tag:privacy-max

# Executar experiments d'un sol proveïdor
python experiment_runner.py experiments.csv --filter provider:openai

# Executar un experiment concret per ID
python experiment_runner.py experiments.csv --filter id:exp_openai_basic

# Dry-run (validar CSV sense executar)
python experiment_runner.py experiments.csv --dry-run

# Amb més detalls de log
python experiment_runner.py experiments.csv --verbose

# Intercalar per proveïdor (round-robin: evita rate limits)
python experiment_runner.py experiments.csv --interleave

# Afegir delay entre experiments (en segons)
python experiment_runner.py experiments.csv --delay 5

# Combinat: intercalar + delay de 3 segons
python experiment_runner.py experiments.csv --interleave --delay 3
```

## Format CSV

El fitxer `experiments.csv` es pot editar directament amb Excel o LibreOffice Calc.

### Columnes

| Columna | Descripció | Exemple |
|---------|------------|---------|
| `id` | Identificador únic | `exp_openai_basic` |
| `name` | Nom descriptiu | `OpenAI GPT-5.1 - Basic` |
| `provider` | Proveïdor LLM | `openai`, `anthropic`, `google` |
| `model` | Model a usar | `gpt-5.1-chat-latest` |
| `temperature` | Temperatura (0.0-2.0) | `1.0` |
| `user_prompt` | Prompt de l'usuari | `Fes-me una PSBT de 3 BTC...` |
| `followup_prompts` | Prompts addicionals (separats per `|`) | `Millora la privacitat|Afegeix més outputs` |
| `repetitions` | Repeticions per estadística | `3` |
| `timeout_seconds` | Timeout per experiment | `300` |
| `network` | Xarxa Bitcoin | `mainnet`, `testnet` |
| `tags` | Tags per filtrar (separats per `|`) | `openai|basic|baseline` |
| `enabled` | Activat/desactivat | `true`, `false` |

### Notes importants

- **Múltiples followups**: Separa'ls amb `|` (pipe)
- **Tags**: Separa'ls amb `|` (pipe)  
- **Textos amb comes**: Envolta'ls amb cometes dobles `"text, amb comes"`
- **Desactivar experiment**: Canvia `enabled` a `false`

## Proveïdors LLM Suportats

| Provider | Models Disponibles |
|----------|-------------------|
| `openai` | gpt-5.1-chat-latest, gpt-4o, gpt-4o-mini, o3, o1 |
| `anthropic` | claude-opus-4-5-20251101, claude-sonnet-4-20250514 |
| `google` | gemini-3-pro-preview, gemini-2.0-flash-exp |

## Resultats

### CSV (Resum)
- `experiment_id`, `experiment_name`
- `llm_provider`, `llm_model`, `llm_temperature`
- `user_prompt`, `success`, `execution_time_seconds`
- `psbt_generated`
- `privacy_score` (0-100)
- `privacy_grade` (A+, A, B, C, D, E, F)

### JSON (Detallat)
Inclou tot l'anterior més:
- `privacy_breakdown` amb tots els factors de puntuació
- Ruta al fitxer PSBT

## Privacy Score

| Factor | Penalització |
|--------|-------------|
| Canvi determinístic (decimal) | -25 punts |
| Canvi optimal input | -15 punts |
| Reutilització d'adreces | -30 punts |
| Linkability / Baixa entropia | -20 punts |
| Múltiples inputs | -10 punts |
| Asimetria tipus adreça | -10 punts |
| Quantitats rodones | -10 punts |
| **Bonus: CoinJoin detectat** | +20 punts |

## Afegir Nous Experiments

1. Obre `experiments.csv` amb Excel/LibreOffice Calc
2. Afegeix una nova fila amb un ID únic
3. Desa el fitxer (format CSV)
4. Valida: `python experiment_runner.py experiments.csv --dry-run`
5. Executa: `python experiment_runner.py experiments.csv --filter id:<nou_id>`

## Requisits

```bash
pip install -r requirements.txt

# Variables d'entorn (.env)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AIza...
XPUB=zpub6...
```

## Anàlisi de Resultats

```python
import pandas as pd

df = pd.read_csv("results/experiments_*.csv")

# Mitjana per proveïdor
df.groupby("llm_provider")["privacy_score"].mean()

# Millor model
df.groupby("llm_model")["privacy_score"].mean().sort_values(ascending=False)
```
