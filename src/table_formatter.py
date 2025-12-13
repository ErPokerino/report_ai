"""
Modulo per formattare tabelle per Quarto.
"""
import pandas as pd
from IPython.display import Markdown, display


def format_table(df, caption=None, digits=3):
    """
    Formatta un DataFrame come tabella Quarto markdown.
    
    Args:
        df: DataFrame da formattare
        caption: Caption opzionale per la tabella
        digits: Numero di cifre decimali per valori numerici
        
    Returns:
        Markdown object da visualizzare
    """
    if df is None or len(df) == 0:
        return Markdown("*Nessun dato disponibile*")
    
    # Arrotonda valori numerici
    df_formatted = df.copy()
    numeric_cols = df_formatted.select_dtypes(include=['float64', 'float32']).columns
    for col in numeric_cols:
        df_formatted[col] = df_formatted[col].round(digits)
    
    # Converti in markdown
    markdown_str = df_formatted.to_markdown(index=False)
    
    if caption:
        markdown_str = f"**{caption}**\n\n{markdown_str}"
    
    return Markdown(markdown_str)


def format_summary_dict(data_dict, title=None):
    """
    Formatta un dizionario di statistiche come tabella.
    
    Args:
        data_dict: Dizionario con chiavi e valori
        title: Titolo opzionale
        
    Returns:
        Markdown object da visualizzare
    """
    df = pd.DataFrame(list(data_dict.items()), columns=['Metrica', 'Valore'])
    
    if title:
        return format_table(df, caption=title)
    return format_table(df)


def display_table(df, caption=None, digits=3):
    """
    Visualizza direttamente una tabella formattata.
    
    Args:
        df: DataFrame da visualizzare
        caption: Caption opzionale
        digits: Numero di cifre decimali
    """
    display(format_table(df, caption, digits))
