"""
Some utility functions for text processing.
"""

def split_text(text, chunk_size, overlap):
    """
    Splits text into a list of chunks of some size with some overlap.
    """
    words = text.split()
    chunks = []

    if overlap > chunk_size:
        overlap = chunk_size
    
    for i in range(0, len(words)-chunk_size+1, chunk_size-overlap):
        chunk = ' '.join(words[i:i+chunk_size])
        chunks.append(chunk)
    
    return chunks

