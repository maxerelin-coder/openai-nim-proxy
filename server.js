// server.js - Clean NVIDIA NIM Proxy for Janitor.AI
const express = require('express');
const cors = require('cors');
const axios = require('axios');
const app = express();
const PORT = process.env.PORT || 3000;

// ====================== CONFIG ======================
const NIM_API_BASE = 'https://integrate.api.nvidia.com/v1';

// 🔥 PUT YOUR API KEY HERE
const NIM_API_KEY = "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";   // ← Replace this

const SHOW_REASONING = true;      // Show <think> tags
const ENABLE_THINKING = true;     // Enable model thinking

// Safe context limit to prevent 400 errors
const MAX_CONTEXT_TOKENS = 120000;
const RESERVED_FOR_OUTPUT = 8000;

// ===================================================

app.use(cors());
app.use(express.json());

// Health check
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'NVIDIA NIM Proxy',
    max_context: MAX_CONTEXT_TOKENS,
    thinking: ENABLE_THINKING,
    show_reasoning: SHOW_REASONING
  });
});

// List models
app.get('/v1/models', (req, res) => {
  res.json({ object: 'list', data: [{ id: 'mistral-large-3', object: 'model' }] });
});

// Main endpoint
app.post('/v1/chat/completions', async (req, res) => {
  try {
    let { model, messages, temperature, max_tokens, stream } = req.body;

    // Model mapping
    let nimModel = model;
    if (model.includes('large-3') || model.includes('mistral-large')) {
      nimModel = 'mistralai/mistral-large-3-675b-instruct-2512';
    } else if (model.includes('nemotron')) {
      nimModel = 'mistralai/mistral-nemotron';
    }

    // === Context Truncation ===
    let processedMessages = [...messages];
    const estimateTokens = (msgs) => 
      msgs.reduce((acc, msg) => acc + Math.ceil((msg.content?.length || 0) / 4) + 10, 0);

    let totalTokens = estimateTokens(processedMessages);

    while (totalTokens > MAX_CONTEXT_TOKENS - RESERVED_FOR_OUTPUT && processedMessages.length > 4) {
      if (processedMessages[1] && processedMessages[1].role !== 'system') {
        processedMessages.splice(1, 1);
      } else {
        processedMessages.splice(2, 1);
      }
      totalTokens = estimateTokens(processedMessages);
    }

    console.log(`[Proxy] Messages: ${messages.length} → ${processedMessages.length} (~${totalTokens} tokens)`);

    // Build request - only pass what Janitor sends
    const nimRequest = {
      model: nimModel,
      messages: processedMessages,
      temperature: temperature ?? 0.82,     // fallback only if Janitor doesn't send it
      max_tokens: max_tokens ?? 1024,
      stream: stream || false,
    };

    if (ENABLE_THINKING) {
      nimRequest.extra_body = {
        chat_template_kwargs: { thinking: true }
      };
    }

    // Call NIM
    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: {
        'Authorization': `Bearer ${NIM_API_KEY}`,
        'Content-Type': 'application/json'
      },
      responseType: stream ? 'stream' : 'json'
    });

    // Streaming handler
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
              if (data.choices?.[0]?.delta && SHOW_REASONING) {
                const delta = data.choices[0].delta;
                if (delta.reasoning_content || delta.thinking) {
                  const think = delta.reasoning_content || delta.thinking || '';
                  if (think) delta.content = `<think>\n${think}\n</think>\n\n${delta.content || ''}`;
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
      response.data.on('error', () => res.end());

    } else {
      // Non-stream response
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
        message: error.response?.data?.error?.message || error.message || 'Internal server error',
        type: 'invalid_request_error'
      }
    });
  }
});

app.listen(PORT, () => {
  console.log(`🚀 NVIDIA NIM Proxy running on http://localhost:${PORT}`);
  console.log(`   Max Context limited to ${MAX_CONTEXT_TOKENS} tokens`);
  console.log(`   Thinking: ${ENABLE_THINKING ? 'ON' : 'OFF'} | Show reasoning: ${SHOW_REASONING ? 'ON' : 'OFF'}`);
});
