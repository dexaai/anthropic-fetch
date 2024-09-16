import {
  type AIChatClient,
  type AIFetchClient,
  type AIFetchRequestOpts,
  type ChatParams,
  type ChatResponse,
  type ChatStreamChunk,
  type ChatStreamParams,
  type ChatStreamResponse,
  createApiInstance,
} from '@dexaai/ai-fetch';
import { type KyInstance, type Options as KyOptions } from 'ky';

import { type Anthropic } from '../anthropic-types/index.js';
import { StreamCompletionChunker } from './streaming.js';

export type ConfigOpts = {
  /**
   * Defaults to process.env['ANTHROPIC_API_KEY'].
   */
  apiKey?: string | null | undefined;

  /**
   * Defaults to process.env['ANTHROPIC_AUTH_TOKEN'].
   */
  authToken?: string | null | undefined;

  /**
   * Override the default base URL for the API, e.g., "https://api.example.com/v2/"
   *
   * Defaults to process.env['ANTHROPIC_BASE_URL'].
   */
  baseUrl?: string | null | undefined;

  /**
   * Override the Anthropic version.
   */
  anthropicVersion?: string | null | undefined;

  /**
   * Optional header to specify the beta version(s) you want to use.
   *
   * If passing in multiple, use a comma separated list without spaces, e.g. beta1,beta2.
   */
  anthropicBetaVersion?: string | null | undefined;

  /**
   * Defaults to 1000.
   */
  defaultMaxTokens?: number | null | undefined;

  /**
   * Options to pass to the underlying fetch library (Ky).
   * @see https://github.com/sindresorhus/ky/tree/main#options
   */
  kyOptions?: KyOptions;
};

export class AnthropicClient implements AIFetchClient, AIChatClient {
  name = 'anthropic';
  api: KyInstance;
  defaultMaxTokens = 1000;

  constructor(opts: ConfigOpts = {}) {
    const process = globalThis.process || { env: {} };
    const apiKey = opts.apiKey || process.env.OPENAI_API_KEY;
    const anthropicVersion = opts.anthropicVersion || '2023-06-01';
    const prefixUrl =
      opts.baseUrl ||
      process.env.ANTHROPIC_BASE_URL ||
      'https://api.anthropic.com';
    this.defaultMaxTokens = opts.defaultMaxTokens || 1000;
    if (!apiKey)
      throw new Error(
        'Missing Anthropic API key. Please provide one in the config or set the ANTHROPIC_API_KEY environment variable.'
      );

    this.api = createApiInstance({
      ...opts.kyOptions,
      prefixUrl,
      headers: {
        ...opts.kyOptions?.headers,
        'User-Agent': 'anthropic-fetch',
        'x-api-key': apiKey,
        'anthropic-version': anthropicVersion,
        'anthropic-beta': opts.anthropicBetaVersion || undefined,
      },
    });
  }

  private getApi(opts?: AIFetchRequestOpts) {
    return opts ? this.api.extend(opts) : this.api;
  }

  private getCompletionUsage(usage: Anthropic.Usage) {
    return {
      prompt_tokens: usage.input_tokens,
      completion_tokens: usage.output_tokens,
      total_tokens: usage.input_tokens + usage.output_tokens,
    };
  }

  private getChatCompletionChoice(
    anthropicResponse: Anthropic.Messages.Message
  ): ChatResponse['choices'][number] {
    const contentBlocks = anthropicResponse.content;
    let tool_calls:
      | ChatResponse['choices'][number]['message']['tool_calls']
      | undefined;

      const firstContentBlock = contentBlocks[0];
      let content: string | null = null;
  
      if (firstContentBlock?.type === 'text') {
        content = firstContentBlock.text;
      }
  
      contentBlocks.slice(1).forEach((contentBlock) => {
        if (contentBlock?.type === 'tool_use') {
          if (!tool_calls) {
            tool_calls = [];
          }
          tool_calls.push({
            type: 'function',
            id: contentBlock.id,
            function: {
              name: contentBlock.name,
              arguments: contentBlock.input as string,
            },
          });
        }
      });

    return {
      index: 0,
      message: {
        role: 'assistant',
        content,
        tool_calls,
        refusal: null,
      },
      logprobs: null,
      finish_reason: tool_calls ? 'tool_calls' : 'stop',
    };
  }

  private convertMessagesToAnthropicMessages(
    messages: ChatParams<
      Anthropic.MessageCreateParamsStreaming['model']
    >['messages']
  ): {
    systemMessage: string | undefined;
    messages: Anthropic.Messages.MessageCreateParamsNonStreaming['messages'];
  } {
    // anthropic doesn't support system messages in the messages param
    const systemMessages = messages.filter(
      (message) => message.role === 'system'
    );
    const systemMessage = systemMessages.length
      ? systemMessages.join('\n')
      : undefined;
    const messagesWithoutSystem = messages.filter(
      (message) => message.role !== 'system'
    );

    const anthropicMessages: Anthropic.Messages.MessageParam[] =
      messagesWithoutSystem.map((message) => {
        let content:
          | string
          | (
              | Anthropic.Messages.TextBlockParam
              | Anthropic.Messages.ImageBlockParam
              | Anthropic.Messages.ToolUseBlockParam
              | Anthropic.Messages.ToolResultBlockParam
            )[];
        switch (message.role) {
          // tool results
          case 'function':
          case 'tool':
            content = [
              {
                type: 'tool_result',
                content: message.content || '',
                // tool use id should be defined for a anthropic tool result.
                tool_use_id: message.tool_call_id || '',
                // todo: expand ChatMessage to include an error filed
                is_error: undefined,
              },
            ];
            return {
              role: 'user',
              content,
            };
          case 'assistant':
            // Tool use
            if (message.tool_calls) {
              // add any text content
              content = message.content
                ? [
                    {
                      type: 'text',
                      text: message.content,
                    },
                  ]
                : [];

              // add any tool uses
              content = content.concat(
                message.tool_calls.map((toolCall) => ({
                  type: 'tool_use',
                  id: toolCall.id,
                  name: toolCall.function.name,
                  input: toolCall.function.arguments,
                }))
              );
            } else {
              // normal assistant message
              content = message.content || '';
            }

            return {
              role: 'assistant',
              content,
            };
          default:
            return {
              role: 'user',
              content: message.content || '',
            };
        }
      });

    return {
      systemMessage,
      messages: anthropicMessages,
    };
  }

  private convertToolsToAnthropicTools(
    tools?: ChatParams<Anthropic.MessageCreateParamsStreaming['model']>['tools']
  ): Anthropic.Messages.Tool[] | undefined {
    return tools?.map((tool) => ({
      name: tool.function.name,
      description: tool.function.description,
      input_schema: tool.function
        .parameters as Anthropic.Messages.Tool.InputSchema,
    }));
  }

  private convertToolChoiceToAnthropicToolChoice(
    toolChoice: ChatParams<
      Anthropic.MessageCreateParamsStreaming['model']
    >['tool_choice']
  ):
    | Anthropic.Messages.MessageCreateParamsNonStreaming['tool_choice']
    | undefined {
    switch (toolChoice) {
      // anthropic doesn't support none
      case 'none':
      case undefined:
        return undefined;
      case 'auto':
        return {
          type: 'auto',
        };
      case 'required':
        return {
          type: 'any', // must use one or more available tools
        };
      default:
        return {
          type: 'tool',
          name: toolChoice.function.name,
        };
    }
  }

  /** Create a message. */
  async createChatCompletion(
    params: ChatParams<Anthropic.MessageCreateParamsStreaming['model']>,
    opts?: AIFetchRequestOpts
  ): Promise<ChatResponse> {
    const { systemMessage, messages } = this.convertMessagesToAnthropicMessages(
      params.messages
    );
    const tools = this.convertToolsToAnthropicTools(params.tools);
    const toolChoice = this.convertToolChoiceToAnthropicToolChoice(
      params.tool_choice
    );
    const anthropicParams: Anthropic.Messages.MessageCreateParamsNonStreaming =
      {
        model: params.model,
        system: systemMessage,
        messages,
        max_tokens: params.max_tokens || this.defaultMaxTokens,
        tools,
        tool_choice: toolChoice,
        top_k: params.n || undefined,
        top_p: params.top_p || undefined,
        temperature: params.temperature || undefined,
        stop_sequences:
          typeof params.stop === 'string'
            ? [params.stop]
            : params.stop || undefined,
      };
    const created = Date.now() / 1000;

    console.log('anthropic params', JSON.stringify(anthropicParams, null, 2));

    const response = await this.getApi(opts)
      .post('messages', { json: anthropicParams })
      .json();
    const anthropicResponse = response as Anthropic.Messages.Message;

    // console.log(anthropicResponse)

    // Convert to the ChatReponse format
    const choice = this.getChatCompletionChoice(anthropicResponse);
    const usage = this.getCompletionUsage(anthropicResponse.usage);
    const chatResponse: ChatResponse = {
      id: anthropicResponse.id,
      object: 'chat.completion',
      model: anthropicResponse.model,
      created,
      usage,
      choices: [choice],
    };

    return chatResponse;
  }

  /** Create a message and stream the response. */
  async streamChatCompletion(
    params: ChatStreamParams<Anthropic.MessageCreateParamsStreaming['model']>,
    opts?: AIFetchRequestOpts
  ): Promise<ChatStreamResponse> {
    // convert the params -> anthropic api format

    const response = await this.getApi(opts).post('messages', {
      json: { ...params, stream: true },
      onDownloadProgress: () => {}, // trick ky to return ReadableStream.
    });
    const stream = response.body as ReadableStream;
    return stream.pipeThrough(
      new StreamCompletionChunker(
        (response: Anthropic.Messages.RawMessageStreamEvent) => {
          // switch (response.type) {
          //   case 'message_start':
          //     return response.message
          //   case 'message_stop':
          //     return response
          //   case 'content_block_start':
          //     return response.content_block
          //   case 'content_block_delta':
          //     switch (response.delta.type) {
          //       case 'text_delta':
          //         return response.delta.text
          //       default:
          //         return response.delta.partial_json
          //     }
          //   case 'content_block_stop':
          //     return response
          //   default:
          //     return response
          // }

          // const chatStreamChunk = {

          // }

          return response as unknown as ChatStreamChunk;
        }
      )
    );
  }
}
