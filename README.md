# openfusion

An open-source, drop-in compound-model proxy. Point any OpenAI-compatible tool at it,
set `model: "openfusion"`, and your prompt is fanned out to a panel of LLMs in parallel —
then a judge model reads every response (consensus, contradictions, blind spots) and streams
back a single synthesized answer that aims to beat any one of them.

It's the open version of the mixture-of-agents idea behind OpenRouter's Fusion: better answers
from models you already pay for, as a tunable, forkable recipe instead of a black box.

## Status

Planning. Architecture is approved (see [DESIGN.md](DESIGN.md)); implementation not started.

## How it will work

```
client (Cursor / OpenAI SDK / anything)
        │  POST /v1/chat/completions   model="openfusion"
        ▼
   openfusion proxy ──► panel member A ┐
                   ──► panel member B ├─ parallel fan-out
                   ──► panel member C ┘
                        │
                        ▼
                   judge model  ──►  streamed synthesized answer (SSE)
```

- **Drop-in.** OpenAI-compatible `POST /v1/chat/completions` + `/v1/models`, real SSE streaming.
- **Default recipe is self-fusion.** Sample one model N times and judge the spread — beats the
  solo model on a single API key, no multi-provider juggling.
- **No lock-in.** Each panel member + judge is `{base_url, api_key, model}`. OpenRouter is the
  default upstream; OpenAI, Together, local vLLM/Ollama all work.
- **Config-driven.** Panel, judge, strategy, and timeouts live in `openfusion.yaml`.

## Stack

Python 3.11+ / FastAPI / httpx / uvicorn.

## License

MIT (intended).
