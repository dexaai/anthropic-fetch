# Anthropic Fetch

[![Build Status](https://github.com/dexaai/anthropic-fetch/actions/workflows/main.yml/badge.svg)](https://github.com/dexaai/anthropic-fetch/actions/workflows/main.yml) [![npm version](https://img.shields.io/npm/v/anthropic-fetch.svg?color=0c0)](https://www.npmjs.com/package/anthropic-fetch)

A minimal and opinionated Anthropic API client built on top of `ai-fetch`.

`anthropic-fetch` provides a streamlined interface for interacting with Anthropic's AI models, leveraging the consistent and minimal approach of `ai-fetch`.

### Key Features:

- Fast and small client that doesn't patch fetch
- Supports all environments with native fetch: Node 18+, browsers, Deno, Cloudflare Workers, etc
- Consistent interface aligned with other `ai-fetch` derived clients
- Focused on chat completions and embeddings for Anthropic models

## Install

```bash
npm install anthropic-fetch
```

This package requires `node >= 18` or an environment with `fetch` support.

This package exports [ESM](https://gist.github.com/sindresorhus/a39789f98801d908bbc7ff3ecc99d99c). If your project uses CommonJS, consider switching to ESM or use the [dynamic `import()`](https://v8.dev/features/dynamic-import) function.

## Usage


```ts
import { AnthropicClient } from 'anthropic-fetch';
const client = new AnthropicClient({
apiKey: 'your-api-key-here',
});
// Generate a chat completion
const response = await client.createChatCompletion({
model: 'claude-3-opus-20240229',
messages: [{ role: 'user', content: 'Hello, Claude!' }],
});
console.log(response.choices[0].message.content);
```

The `apiKey` is optional and will be read from `process.env.OPENAI_API_KEY` if present.

## API

The Anthropic Fetch API implements the following `ai-fetch` interfaces

```ts
// Generate a single chat completion
client.createChatCompletion(params: ChatParams): Promise<ChatResponse>;

// Stream a single completion via a ReadableStream
client.streamChatCompletion(params: ChatStreamParams): Promise<ChatStreamResponse>;
```

### Type Definitions

The type definitions are available through TSServer, and can be found in the source code.

## Derived from AI Fetch

`anthropic-fetch` is built on top of `ai-fetch`

## License

MIT © [Dexa](https://dexa.ai)
