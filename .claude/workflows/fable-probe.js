export const meta = {
  name: 'fable-probe',
  description: 'Medical many-shot leave-one-out probe: one tool-less agent per conversation answers its prompt file',
  whenToUse: 'After bench/llm/probe_all.py dump has written the prompt files. args = {dir, stems}. Returns {stem, out} per conversation; write each out to bench/results/probe/<run>/<stem>.json, then probe_all.py score.',
  phases: [{ title: 'Answer', detail: 'one agent per conversation, Read on its own prompt file only' }],
}

// args.dir: absolute folder of the prompt files; args.stems: the conversation stems to answer.
// Each agent reads ONE file and answers from it. No other file, no search, no shell: the repository
// holds the answer key (data/question_train.csv, bench/mine/), and the measurement is only honest if
// the agent never sees it. The main session verifies the tool calls in the transcripts afterwards.
const SCHEMA = {
  type: 'object',
  properties: {
    answers: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          q: { type: 'integer' },
          quote: { type: 'string' },
          answer: { type: 'string', enum: ['yes', 'no'] },
          segments: { type: 'array', items: { type: 'integer' } },
        },
        required: ['q', 'quote', 'answer', 'segments'],
      },
    },
  },
  required: ['answers'],
}

const dir = args.dir
const stems = args.stems
phase('Answer')
log(`${stems.length} conversations, one agent each`)
const results = await parallel(stems.map(stem => () =>
  agent(
    `You are answering a benchmark prompt that lives in one file. Read that file completely and nothing else.\n\n` +
    `File: ${dir}/${stem}.txt\n\n` +
    `It has about 4,000 to 5,000 lines: a "### SYSTEM" section (the instructions), then many pairs of "### EXAMPLE INPUT" ` +
    `and "### EXAMPLE OUTPUT" (worked consultations answered exactly the way the annotators did), then one "### INPUT" ` +
    `section with the transcript and the ten questions you must answer. The Read tool returns at most 2000 lines per ` +
    `call, so read it in pieces (offset 1, then 2001, then 4001, and so on) until a call returns fewer lines than you asked for.\n\n` +
    `Rules that make this measurement honest:\n` +
    `- Use the Read tool on that one file only. Do not open, search, list or read any other file or folder, do not run ` +
    `any command, do not browse. The repository contains the answer key and you must never see it.\n` +
    `- Answer from the SYSTEM instructions, the examples and the INPUT alone. Follow the SYSTEM section literally: ` +
    `near-misses are "no", quote an utterance verbatim, cite the unit ids the way the examples do.\n` +
    `- Give exactly ten answers, q = 1 to 10, in order, through the StructuredOutput tool. Nothing else.`,
    args.model ? { label: stem, phase: 'Answer', schema: SCHEMA, model: args.model } : { label: stem, phase: 'Answer', schema: SCHEMA },
  ).then(out => ({ stem, out })),
))
const done = results.filter(Boolean)
log(`${done.length} of ${stems.length} answered`)
return done
