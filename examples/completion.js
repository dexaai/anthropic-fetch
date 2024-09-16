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

async function testChatCompletion() {
  console.log('Testing Anthropic...');
  let messages = [
    { role: 'user', content: 'Hello, Claude! What is the weather in Tokyo? ' },
  ];
  const anthropicResponse = await anthropicClient.createChatCompletion({
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
  console.log('Anthropic Response:', anthropicResponse.choices[0].message);

  messages.push(anthropicResponse.choices[0].message);

  if (anthropicResponse.choices[0].message.tool_calls) {
    console.log(
      'Function Call:',
      anthropicResponse.choices[0].message.tool_calls
    );
    for (const toolCall of anthropicResponse.choices[0].message.tool_calls) {
      const toolUse = get_current_weather(toolCall.function.arguments);
      messages.push({
        role: 'tool',
        tool_call_id: toolCall.id,
        content: JSON.stringify(toolUse),
      });
    }

    console.log('Messages:', messages);
    const anthropicResponse2 = await anthropicClient.createChatCompletion({
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

    console.log('Anthropic Response 2:', anthropicResponse2.choices[0].message);
  }
}

testChatCompletion();