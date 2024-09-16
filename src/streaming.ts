/** A function that converts from raw Completion response from Anthropic
 * into a nicer object which includes the first choice in response from Anthropic.
 */
type ResponseFactory<Raw, Nice> = (response: Raw) => Nice;

/**
 * A parser for the streaming responses from the Anthropic API.
 *
 * Conveniently shaped like an argument for WritableStream constructor.
 */
class AnthropicStreamParser<Raw, Nice> {
  private responseFactory: ResponseFactory<Raw, Nice>;
  onchunk?: (chunk: Nice) => void;
  onend?: () => void;
  buffer: string;

  constructor(responseFactory: ResponseFactory<Raw, Nice>) {
    this.responseFactory = responseFactory;
    this.buffer = '';
  }

  /**
   * Takes the ReadableStream chunks, produced by `fetch` and turns them into
   * `CompletionResponse` objects.
   * @param chunk The chunk of data from the stream.
   */
  write(chunk: Uint8Array): void {
    const decoder = new TextDecoder();
    const s = decoder.decode(chunk);
    let parts = s.split('\n\n');

     // Handle buffering
    if (this.buffer) {
      parts[0] = this.buffer + parts[0];
      this.buffer = '';
    }
    if (!s.endsWith('\n\n')) {
      this.buffer = parts.pop() || '';
    }

    parts.forEach((part) => {
      const lines = part.split('\n');
      const eventType = lines[0]?.substring(7); // Remove "event: "
      const data = JSON.parse(lines[1]?.substring(6) || '{}'); // Remove "data: " and parse JSON

      if (eventType === 'content_block_delta' && data.delta.type === 'text_delta') {
        try {
          const niceChunk = this.responseFactory(data.delta as Raw);
          this.onchunk?.(niceChunk);
        } catch (error) {
          console.error('Error processing chunk:', error);
        }
      } else if (eventType === 'message_stop') {
        this.onend?.();
      }
    });
  }
}

/**
 * A transform stream that takes the streaming responses from the Anthropic API
 * and turns them into useful response objects.
 */
export class StreamCompletionChunker<Raw, Nice>
  implements TransformStream<Uint8Array, Nice>
{
  writable: WritableStream<Uint8Array>;
  readable: ReadableStream<Nice>;

  constructor(responseFactory: ResponseFactory<Raw, Nice>) {
    const parser = new AnthropicStreamParser(responseFactory);
    this.writable = new WritableStream(parser);
    this.readable = new ReadableStream({
      start(controller) {
        parser.onchunk = (chunk: Nice) => controller.enqueue(chunk);
        parser.onend = () => controller.close();
      },
    });
  }
}