Place trace JSON exports (schema `bef_trace_v1`) here.
Each file should look like:
{
  "schema": "bef_trace_v1",
  "trace_id": "trace_001",
  "field_modulus": 2147483647,
  "num_steps": 12345,
  "vector_length": 500000,
  "chunk_length": 100000,
  "chunks": [
     {"chunk_index":0, "offset":0, "values": [...]},
     ...
  ]
}
