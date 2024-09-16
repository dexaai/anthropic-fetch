import { AnthropicClient } from '../dist/anthropic-client.js';
import dotenv from 'dotenv';

dotenv.config(
);

const get_current_weather = (args) => {
  return {
    result: `70 degrees.`,
  };
};

const anthropicClient = new AnthropicClient({
  baseUrl: 'https://api.anthropic.com/v1',
});


const parseStream = async (stream) => {
  // Keep track of the stream's output
  let chunk = {};

  // Get a reader from the stream
  const reader = stream.getReader();

  while (true) {
    const { done, value } = await reader.read();

    if (done) {
      // If the stream is done, break out of the loop and save the conversation
      // to the cache before returning.
      break;
    }

    // Create the initial chunk
    if (!chunk.id) {
      chunk = value;
    }

    const delta = value.choices[0].delta;

    if (Object.keys(chunk).length === 0) {
      chunk = value;
    }

    // Send an update to the caller
    const messageContent = delta?.content;
    if (typeof messageContent === 'string') {
      try {
        process.stdout.write(messageContent);
      } catch (err) {
        console.error('Error handling update', err);
      }
    }

    // Merge the delta into the chunk
    const { content, function_call, tool_calls } = delta;
    if (content) {
      chunk.choices[0].delta.content = `${chunk.choices[0].delta.content}${content}`;
    }
    if (function_call) {
      const existingFunctionCall = chunk.choices[0].delta.function_call;
      chunk.choices[0].delta.function_call = {
        ...existingFunctionCall,
        arguments: `${existingFunctionCall?.arguments ?? ''}${function_call.arguments}`,
      };
    }
    if (tool_calls) {
      const existingToolCalls = chunk.choices[0].delta.tool_calls;
      if (!existingToolCalls) {
        chunk.choices[0].delta.tool_calls = tool_calls;
      } else {
        chunk.choices[0].delta.tool_calls = existingToolCalls.map(
          (existingToolCall) => {
            const matchingToolCall = tool_calls.find(
              (toolCall) => toolCall.index === existingToolCall.index
            );
            if (!matchingToolCall) return existingToolCall;
            const existingArgs = existingToolCall.function?.arguments ?? '';
            const matchingArgs = matchingToolCall?.function?.arguments ?? '';
            return {
              ...existingToolCall,
              function: {
                ...existingToolCall.function,
                arguments: `${existingArgs}${matchingArgs}`,
              },
            };
          }
        );
      }
    }
  }

  // Once the stream is done, release the reader
  reader.releaseLock();

  const choice = chunk.choices[0];
  const response = {
    ...chunk,
    object: 'chat.completion',
    choices: [
      {
        finish_reason: choice.finish_reason,
        index: choice.index,
        message: choice.delta,
        logprobs: choice.logprobs || null,
      },
    ],
  };

  return response;
};


async function testAnthropicStreaming() {
  let messages = [{ role: 'user', content: 'What is the weather in Tokyo?' }];
  console.log('\nTesting Anthropic Streaming...');
  const anthropicStream = await anthropicClient.streamChatCompletion({
    model: 'claude-3-5-sonnet-20240620',
    messages,
    tools: [
      {
        type: 'function',
        function: {
          name: 'get_current_weather',
          description: 'Get the current weather',
          parameters: {
            type: 'object',
            properties: {
              location: { type: 'string' },
            },
            required: ['location'],
          },
        },
      },
    ],
  });

  const response = await parseStream(anthropicStream);
  // console.log('Anthropic Response:', JSON.stringify(response, null, 2)); 
  let toolCalls = response.choices[0].message.tool_calls;
  let toolCallResults = [];
  for (const toolCall of toolCalls) {
    if (toolCall.function.name === 'get_current_weather') {
      // note: must parse the arguments as JSON, before adding to the messages
      toolCall.function.arguments = JSON.parse(toolCall.function.arguments);
      const toolUse = get_current_weather(toolCall.function.arguments);
      toolCallResults.push({
        role: 'tool',
        tool_call_id: toolCall.id,
        content: JSON.stringify(toolUse),
      });
    }
  }

  // add the tool calls and results to the messages
  messages.push({
    ...response.choices[0].message,
    tool_calls: toolCalls,
  });

  // add the tool call results to the messages
  messages = messages.concat(toolCallResults);

  const anthropicStream2 = await anthropicClient.streamChatCompletion({
    model: 'claude-3-5-sonnet-20240620',
    messages,
    tools: [
      {
        type: 'function',
        function: {
          name: 'get_current_weather',
          description: 'Get the current weather',
          parameters: {
            type: 'object',
            properties: {
              location: { type: 'string' },
            },
            required: ['location'],
          },
        },
      },
    ],
  });

  await parseStream(anthropicStream2);
}

testAnthropicStreaming();

