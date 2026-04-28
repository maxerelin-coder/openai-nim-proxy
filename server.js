// server.js - Optimized OpenAI to NVIDIA NIM Proxy for Janitor.AI
const express = require('express');
const cors = require('cors');
const axios = require('axios');
const app = express();
const PORT = process.env.PORT || 3000;

// ====================== CONFIG ======================
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

const SHOW_REASONING = true;          // Set to true to show thinking in <think> tags
const ENABLE_THINKING = true;         // Global thinking toggle

// Safe max context for Mistral Large 3 / Nemotron on NIM (prevents 400 errors)
const MAX_CONTEXT_TOKENS = 120000;    // You can raise to 150000-200000 if you want more aggressive context
const RESERVED_FOR_OUTPUT = 8000;     // Reserve tokens for the model's reply

// ===================================================

app.use(cors());
app.use(express.json());

// Health check
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'NVIDIA NIM Proxy (Optimized for Janitor.AI)',
    max_context: MAX_CONTEXT_TOKENS,
    thinking: ENABLE_THINKING,
    show_reasoning: SHOW_REASONING
  });
});

// Simple models list
app.get('/v1/models', (req, res) => {
  res.json({
    object: 'list',
    data: [{ id: 'mistral-large-3', object: 'model', owned_by: 'nvidia-nim-proxy' }]
  });
});

// Main chat completions endpoint
app.post('/v1/chat/completions', async (req, res) => {
  try {
    let { model, messages, temperature, max_tokens, stream } = req.body;

    // --- Smart Model Mapping ---
    let nimModel = model;
    if (model.includes('large-3') || model.includes('mistral-large')) {
      nimModel = 'mistralai/mistral-large-3-675b-instruct-2512';
    } else if (model.includes('nemotron')) {
      nimModel = 'mistralai/mistral-nemotron';
    }

    // --- Context Truncation (Most Important Fix) ---
    let processedMessages = [...messages];

    // Rough token estimation (1 token ≈ 4 characters)
    const estimateTokens = (msgs) => {
      return msgs.reduce((acc, msg) => acc + Math.ceil((msg.content?.length || 0) / 4) + 10, 0);
    };

    let totalTokens = estimateTokens(processedMessages);

    // Truncate oldest messages (keep system + newest user/assistant turns)
    while (totalTokens > MAX_CONTEXT_TOKENS - RESERVED_FOR_OUTPUT && processedMessages.length > 4) {
      // Remove the second message (first is usually system, don't remove it)
      if (processedMessages[1] && processedMessages[1].role !== 'system') {
        processedMessages.splice(1, 1);
      } else {
        processedMessages.splice(2, 1); // fallback
      }
      totalTokens = estimateTokens(processedMessages);
    }

    console.log(`[Proxy] Original messages: ${messages.length} | After truncation: ${processedMessages.length} (~${totalTokens} tokens)`);

    // --- Build NIM Request ---
    const nimRequest = {
      model: nimModel,
      messages: processedMessages,
      temperature: temperature || 0.82,
      max_tokens: max_tokens || 1024,
      stream: stream || false,
    };

    // Enable thinking (works on many Nemotron / Mistral variants on NIM)
    if (ENABLE_THINKING) {
      nimRequest.extra_body = {
        chat_template_kwargs: {
          thinking: true,
          // enable_thinking: true,     // Uncomment if thinking: true doesn't work
        }
      };
    }

    // --- Call NVIDIA NIM ---
    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: {
        'Authorization': `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      responseType: stream ? 'stream' : 'json'
    });

    // Streaming handler with reasoning support
    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      let buffer = '';

      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            if (line.includes('[DONE]')) {
              res.write('data: [DONE]\n\n');
              continue;
            }

            try {
              const data = JSON.parse(line.slice(6));
              if (data.choices?.[0]?.delta) {
                const delta = data.choices[0].delta;

                if (SHOW_REASONING && (delta.reasoning_content || delta.thinking)) {
                  const thinking = delta.reasoning_content || delta.thinking || '';
                  if (thinking) {
                    delta.content = `<think>\n${thinking}\n</think>\n\n${delta.content || ''}`;
                  }
                  delete delta.reasoning_content;
                  delete delta.thinking;
                }
              }
              res.write(`data: ${JSON.stringify(data)}\n\n`);
            } catch (e) {
              res.write(line + '\n');
            }
          }
        }
      });

      response.data.on('end', () => res.end());
      response.data.on('error', (err) => {
        console.error('Stream error:', err);
        res.end();
      });

    } else {
      // Non-streaming response
      const openaiResponse = {
        id: `chatcmpl-${Date.now()}`,
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: model,
        choices: response.data.choices.map(choice => ({
          index: choice.index,
          message: {
            role: choice.message.role,
            content: SHOW_REASONING && choice.message.reasoning_content
              ? `<think>\n${choice.message.reasoning_content}\n</think>\n\n${choice.message.content || ''}`
              : choice.message.content || ''
          },
          finish_reason: choice.finish_reason
        })),
        usage: response.data.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
      };

      res.json(openaiResponse);
    }

  } catch (error) {
    console.error('Proxy Error:', error.response?.data || error.message);
    res.status(error.response?.status || 500).json({
      error: {
        message: error.response?.data?.error?.message || error.message,
        type: 'invalid_request_error',
        code: error.response?.status || 500
      }
    });
  }
});

app.listen(PORT, () => {
  console.log(`🚀 NVIDIA NIM Proxy running on http://localhost:${PORT}`);
  console.log(`   Max context limited to ${MAX_CONTEXT_TOKENS} tokens`);
  console.log(`   Thinking mode: ${ENABLE_THINKING ? 'ENABLED' : 'DISABLED'}`);
  console.log(`   Show reasoning: ${SHOW_REASONING ? 'ENABLED' : 'DISABLED'}`);
});
