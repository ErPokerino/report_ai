# Sistema di Reportistica Automatica con Quarto e AI

Sistema per la generazione automatica di reportistica usando Quarto per output multi-format (HTML, PDF, PPT) e AI (LangChain) per analisi testuali, commenti ai grafici e generazione di contenuti.

## Documentazione

- **[DOCS.md](DOCS.md)**: Documentazione funzionale completa che spiega come funziona il sistema, il flusso dati end-to-end, i componenti principali e le interazioni tra moduli. **Consigliata per comprendere l'architettura del sistema.**
- Questo README: Focus su setup, installazione e uso pratico.

## Caratteristiche

- 📊 Generazione automatica di visualizzazioni
- 🤖 Analisi AI con LangChain
- 📄 Output multi-format (HTML, PDF, RevealJS)
- 🐍 Ambiente Python gestito con UV
- 📝 Tabelle formattate con Quarto
- 🎨 Commenti AI strutturati con markdown

## Prerequisiti

- [Quarto](https://quarto.org/docs/get-started/) installato
- [UV](https://github.com/astral-sh/uv) installato
- Python 3.10 o superiore
- API key OpenAI (per funzionalità AI)

## Setup

1. Clona o naviga nella directory del progetto

2. Installa le dipendenze con UV:
```bash
uv sync
```

3. Configura la API key OpenAI:
   
   **Metodo consigliato (file .env):**
   - Copia `.env.example` in `.env`
   - Apri il file `.env` nella root del progetto
   - Sostituisci `your-api-key-here` con la tua API key OpenAI:
     ```
     OPENAI_API_KEY=sk-tua-chiave-qui
     ```
   - Il file `.env` viene caricato automaticamente dai moduli AI
   
   **Metodo alternativo (variabile d'ambiente):**
   ```bash
   # Windows PowerShell
   $env:OPENAI_API_KEY="your-api-key-here"
   
   # Linux/Mac
   export OPENAI_API_KEY="your-api-key-here"
   ```

4. Attiva l'ambiente virtuale (opzionale, UV gestisce automaticamente):
```bash
# Windows
.venv\Scripts\Activate.ps1

# Linux/Mac
source .venv/bin/activate
```

## Uso

### Generare un report

#### Metodo 1: Script Python (Consigliato)

```bash
# Genera solo HTML
python scripts/generate_report.py --format html

# Genera tutti i formati
python scripts/generate_report.py --format all

# Genera PDF
python scripts/generate_report.py --format pdf

# Genera RevealJS (presentazione)
python scripts/generate_report.py --format revealjs

# Usa un report personalizzato
python scripts/generate_report.py --format html --input reports/custom_report.qmd
```

#### Metodo 2: Quarto CLI diretto

```bash
# Report Lucy (dati reali)
quarto render reports/report_lucy.qmd --to html
quarto render reports/report_lucy.qmd --to pdf
```

### Parametri Quarto

Il report supporta parametri personalizzabili:

```bash
# Disabilita AI
quarto render reports/report_lucy.qmd -P enable_ai:false

# Usa dataset personalizzato
quarto render reports/report_lucy.qmd -P dataset_path:"../data/mydata.csv"
```

Oppure modifica direttamente i parametri nel YAML header del file `.qmd`.

## Struttura del Progetto

```
quarto/
├── README.md                    # Questo file (setup e uso)
├── DOCS.md                      # Documentazione funzionale completa
├── pyproject.toml              # Dipendenze Python (UV)
├── uv.lock                     # Lock file UV
├── .env.example                # Template variabili ambiente
├── .gitignore                  # Git ignore
├── .python-version             # Versione Python
├── _quarto.yml                 # Configurazione Quarto globale
├── styles.css                  # CSS personalizzato
│
├── reports/                    # Report Quarto
│   └── report_lucy.qmd         # Report principale Lucy
│
├── src/                        # Codice Python
│   ├── data_loader.py          # Caricamento e preparazione dati
│   ├── lucy_visualizations.py # Visualizzazioni specifiche Lucy
│   ├── ai_analysis.py          # Integrazione LangChain per analisi AI
│   ├── table_formatter.py      # Formattazione tabelle Quarto
│   └── context_loader.py      # Caricamento file di contesto
│
├── data/                       # Dati (dinamici)
│   └── lucy_data.csv           # Dataset Lucy (può variare nel tempo)
│
├── context/                     # File di contesto (dinamici)
│   ├── README.md               # Spiegazione uso cartella contesto
│   └── [altri file di contesto] # File MD, PDF, TXT, etc. (dinamici)
│
└── scripts/                     # Script di utilità
    └── generate_report.py      # Script generazione report
```

## Cartella Context

La cartella `context/` contiene file di contesto dinamici che forniscono informazioni sul progetto, sui dati e sulla conoscenza di dominio specifica.

### Scopo

I file in questa cartella servono a:

- **Fornire contesto al sistema AI**: I file possono essere referenziati nei prompt AI per migliorare la qualità delle analisi, utilizzare terminologia corretta e fornire conoscenza di dominio specifica.

- **Documentare il progetto**: Informazioni sul sistema, sui dati, sui flussi di processo (es. flusso invoice, machine learning pipeline) e sul dominio di applicazione.

- **Supportare decisioni analitiche**: Conoscenza di dominio che aiuta a decidere quali analisi effettuare e come interpretare i risultati.

### Tipi di File Supportati

La cartella può contenere vari formati:

- **File Markdown (.md)**: Documentazione testuale, dizionari dati, conoscenza di dominio
- **File PDF**: Documentazione formale, specifiche tecniche, diagrammi di flusso
- **File di testo (.txt)**: Note, annotazioni, informazioni aggiuntive
- **Altri formati**: Qualsiasi file che fornisce contesto utile al progetto

### Note Importanti

- **I file sono dinamici**: Possono essere aggiunti, modificati o rimossi in base alle esigenze del progetto
- **Non committare informazioni sensibili**: Verificare sempre il contenuto prima di committare
- **Manutenzione**: Mantenere i file aggiornati con le informazioni più recenti

Per maggiori dettagli, consulta `context/README.md`.

## Report Lucy

Il report `reports/report_lucy.qmd` analizza i dati reali del sistema Lucy (applicativo documentale Luxottica).

### Contenuti del Report

- Analisi delle predizioni di riconoscimento documentale
- Analisi performance: Precision, Recall, F1, Accuracy per metodo
- Confronto ML vs Query-based
- Analisi errori: Pattern FP/FN
- Analisi temporale e geografica
- Commenti AI strutturati con elenchi puntati e formattazione markdown

### Dati Lucy

I dati si trovano in `data/lucy_data.csv`. **Nota**: Il dataset è dinamico e può cambiare nel tempo.

**Campi tipici includono**:
- Campo analizzato: `id_subject` (codice fornitore) e possibilmente altri campi
- Predizioni vs verità (quando validato)
- Metodo utilizzato per ogni predizione
- Confidence score
- Timestamp delle predizioni
- Autovalidazione: flag che indica se il sistema ha autovalidato il documento

Per dettagli completi sulle colonne disponibili, consulta `context/data_dictionary.md`.

## Tecnologie

- **Quarto**: Rendering documenti multi-format
- **Python**: Linguaggio principale
- **UV**: Gestione dipendenze Python
- **LangChain**: Framework AI per analisi e generazione contenuti
- **Pandas/Matplotlib/Seaborn**: Analisi dati e visualizzazioni
- **Tabulate**: Formattazione tabelle markdown

## Configurazione AI

Per abilitare le funzionalità AI (LangChain), configura la API key:

**Metodo consigliato:** Modifica il file `.env` nella root del progetto e inserisci le tue API keys:
```
OPENAI_API_KEY=sk-tua-chiave-qui
GOOGLE_API_KEY=tua-chiave-google-qui
```

Il file `.env` viene caricato automaticamente. **Nota**: 
- `OPENAI_API_KEY` è richiesta per i modelli OpenAI (GPT-5.2, GPT-4o)
- `GOOGLE_API_KEY` è opzionale ma consigliata per abilitare i modelli Gemini come fallback
- **Non committare il file `.env`** (è già nel `.gitignore`)

**Metodo alternativo:** Usa variabili d'ambiente di sistema (vedi sezione Setup).

### Modello LLM

Il sistema utilizza una gerarchia di modelli LLM con fallback automatico. L'ordine di fallback è: **GPT-5.2 → Gemini 3 PRO → Gemini Flash 2.5 → GPT-4o**. Se un modello non è disponibile o ha la quota esaurita, il sistema prova automaticamente il modello successivo nella gerarchia.

Il sistema traccia automaticamente quale modello viene utilizzato per ogni chiamata e mostra statistiche di utilizzo nella sezione "Modelli LLM Utilizzati" del report generato.

## Note

- Per usare le funzionalità AI, è necessaria una API key OpenAI valida
- I grafici vengono generati automaticamente dai dati
- Il sistema supporta parametri Quarto per personalizzazione
- Lo script `scripts/generate_report.py` facilita la generazione automatica in batch
- I file generati vengono salvati nella directory root (esclusi da git tramite `.gitignore`)
- I commenti AI utilizzano markdown per una migliore leggibilità (elenchi, corsivo, grassetto)

## Licenza

MIT
