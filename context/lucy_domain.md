# Conoscenza di Dominio - Sistema Lucy

## Panoramica

Il sistema **Lucy** è un applicativo documentale sviluppato per **Luxottica** per la gestione automatica delle fatture. Il sistema utilizza algoritmi di riconoscimento automatico per estrarre informazioni dalle fatture, in particolare il campo `id_subject` (codice fornitore).


## Terminologia

- **`id_subject`**: Codice identificativo del fornitore 
- **`autovalidated`**: Flag che indica se il sistema ritiene che il documento possa essere autovalidato
- **`id_company`**: Identificativo dell'azienda Luxottica
- **`protocol`**: Numero di protocollo univoco della fattura
- **`prediction`**: Valore predetto dal sistema
- **`truth`**: Valore reale/validato manualmente
- **`comparison`**: Confronto tra predizione e verità (TP, FP, FN, TN)
- **`confidence`**: Livello di confidenza della predizione (0-1)
- **`method_pred`**: Metodo utilizzato per la predizione
- **`is_validated`**: Flag che indica se il record è stato validato manualmente


## Contesto Aziendale

- **Azienda**: Luxottica
- **Tipo documento**: Fatture
- **Validazione**: Processo manuale di verifica delle predizioni

## Note Operative

- Il sistema processa fatture da diversi paesi (campo `country`)
- I metodi vengono utilizzati in modo gerarchico/fallback
- La confidence viene utilizzata per decisioni operative
