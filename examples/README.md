# Example data formats

Five hand-written tickets, the same data in both formats. Real calibration needs at least 500.

- `tickets.jsonl`: one System One request per line, with a `label` on each question (Kev's format).
- `tickets.csv` + `questions.json`: a spreadsheet. Columns named after a question are its labels;
  the other columns (`subject`, `body`) become the state.

```bash
howsure run tickets.jsonl --url http://localhost:8000
howsure run tickets.csv --questions questions.json --url http://localhost:8000
```
