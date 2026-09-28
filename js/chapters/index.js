// LLMClear's chapters, in reading order.
import next from './next.js';
import tokens from './tokens.js';
import embed from './embed.js';
import attention from './attention.js';
import scale from './scale.js';
import use from './use.js';

export const BOX = { slug: 'llmclear', title: 'LLMClear' };
export const CHAPTERS = [next, tokens, embed, attention, scale, use];
