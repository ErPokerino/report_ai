"""
Modulo per formattare tabelle per Quarto.
"""
import pandas as pd
from typing import Optional, Dict, Any, Union
from IPython.display import Markdown, display

# Costanti
DEFAULT_DIGITS = 3
EMPTY_DATA_MESSAGE = "*Nessun dato disponibile*"


def format_table(
    df: Optional[pd.DataFrame],
    caption: Optional[str] = None,
    digits: int = DEFAULT_DIGITS,
    return_string: bool = False
) -> Union[Markdown, str]:
    """
    Formatta un DataFrame come tabella Quarto markdown.
    
    Args:
        df: DataFrame da formattare
        caption: Caption opzionale per la tabella
        digits: Numero di cifre decimali per valori numerici
        return_string: Se True, restituisce una stringa invece di Markdown object
        
    Returns:
        Markdown object o stringa da visualizzare
    """
    if df is None or len(df) == 0:
        return EMPTY_DATA_MESSAGE if return_string else Markdown(EMPTY_DATA_MESSAGE)
    
    # Arrotonda valori numerici
    df_formatted = df.copy()
    numeric_cols = df_formatted.select_dtypes(include=['float64', 'float32']).columns
    for col in numeric_cols:
        df_formatted[col] = df_formatted[col].round(digits)
    
    # Converti in markdown
    markdown_str = df_formatted.to_markdown(index=False)
    
    if caption:
        markdown_str = f"**{caption}**\n\n{markdown_str}"
    
    return markdown_str if return_string else Markdown(markdown_str)


def format_summary_dict(
    data_dict: Dict[str, Any],
    title: Optional[str] = None,
    return_string: bool = False
) -> Union[Markdown, str]:
    """
    Formatta un dizionario di statistiche come tabella.
    
    Args:
        data_dict: Dizionario con chiavi e valori
        title: Titolo opzionale
        return_string: Se True, restituisce una stringa invece di Markdown object
        
    Returns:
        Markdown object o stringa da visualizzare
    """
    df = pd.DataFrame(list(data_dict.items()), columns=['Metrica', 'Valore'])
    return format_table(df, caption=title, return_string=return_string)


def display_table(
    df: pd.DataFrame,
    caption: Optional[str] = None,
    digits: int = DEFAULT_DIGITS
) -> None:
    """
    Visualizza direttamente una tabella formattata.
    
    Args:
        df: DataFrame da visualizzare
        caption: Caption opzionale
        digits: Numero di cifre decimali
    """
    display(format_table(df, caption, digits))
