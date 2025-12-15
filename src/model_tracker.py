"""
Modulo per il tracking globale delle chiamate LLM utilizzate durante la generazione del report.
Usa pattern Singleton per garantire persistenza tra celle Quarto.
"""
from typing import Dict, List, Optional
from collections import Counter


class ModelTracker:
    """
    Tracker globale per registrare le chiamate LLM e determinare quale modello
    è stato effettivamente utilizzato durante la generazione del report.
    
    Usa pattern Singleton per garantire che la stessa istanza sia condivisa
    tra tutte le celle Quarto durante l'esecuzione del report.
    """
    _instance = None
    _calls: List[Dict[str, any]] = []
    
    def __new__(cls):
        """Pattern Singleton: restituisce sempre la stessa istanza."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._calls = []
        return cls._instance
    
    def track_call(self, model_name: str, success: bool = True) -> None:
        """
        Registra una chiamata LLM.
        
        Args:
            model_name: Nome del modello utilizzato (es. "gpt-5.2", "gemini-3-pro-preview")
            success: Se True, la chiamata è riuscita; se False, è fallita
        """
        self._calls.append({
            "model": model_name,
            "success": success
        })
    
    def get_usage_stats(self) -> Dict[str, int]:
        """
        Restituisce statistiche di utilizzo: conteggio chiamate per modello.
        
        Returns:
            Dizionario con modello come chiave e numero di chiamate come valore
        """
        if not self._calls:
            return {}
        
        # Conta solo chiamate riuscite
        successful_calls = [call["model"] for call in self._calls if call.get("success", True)]
        return dict(Counter(successful_calls))
    
    def get_primary_model(self) -> Optional[str]:
        """
        Restituisce il modello più utilizzato (modello primario).
        
        Returns:
            Nome del modello più utilizzato, o None se non ci sono chiamate
        """
        stats = self.get_usage_stats()
        if not stats:
            return None
        
        # Restituisce il modello con più chiamate
        return max(stats.items(), key=lambda x: x[1])[0]
    
    def get_all_calls(self) -> List[Dict[str, any]]:
        """
        Restituisce tutte le chiamate registrate (per debug/analisi).
        
        Returns:
            Lista di dizionari con dettagli di ogni chiamata
        """
        return self._calls.copy()
    
    def reset(self) -> None:
        """
        Resetta il tracker (utile per testing o reset manuale).
        """
        self._calls = []
    
    def get_total_calls(self) -> int:
        """
        Restituisce il numero totale di chiamate registrate.
        
        Returns:
            Numero totale di chiamate
        """
        return len(self._calls)
    
    def get_successful_calls_count(self) -> int:
        """
        Restituisce il numero di chiamate riuscite.
        
        Returns:
            Numero di chiamate riuscite
        """
        return sum(1 for call in self._calls if call.get("success", True))
