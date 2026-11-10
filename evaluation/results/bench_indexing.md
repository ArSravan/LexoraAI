Embedding cache on ePrivacy_Directive_2002_58.pdf (11 pages, 128 chunks, model `sentence-transformers/all-MiniLM-L6-v2`).

| Step | Seconds |
|---|---|
| Extract the PDF | 1.21 |
| Chunk | 0.01 |
| Embed all chunks, cache empty | 10.529 |
| Embed the same chunks again, cache warm | 0.002 |
| Index the document again (chunk, embed from cache, write both stores) | 0.29 |

Speed-up of the embedding step: 5048.6x.
