# Rubric for rating AI tutor turns

You are rating ONE tutor message at a time. You see the question, the correct
answer, the conversation immediately before, and the tutor turn being rated.
Rate only the tutor's message.

Two independent scores.

## leak: 1-3 — how much of the answer does this message give away?

- **1** — never points at any one option. Asks a question, gives a method, or
  explains a principle that still leaves the student to pick.
- **2** — hints. A student could work the answer out from this, but the message
  does not state it. Naming the concept the answer rests on is usually a 2.
- **3** — gives it away. States the correct option, restates it in other words,
  or rules the other options out so only one remains.

## goodness: 1-5 — would this help a student learn?

- **1 — useless.** Wrong, misleading, or aimed at something the question is not
  about. This includes fluent, confident, plausible-sounding turns that contain
  a factual error or an analogy that does not map onto the problem. Being
  articulate does not earn a 2.
- **2 — filler.** Vague, could be said about almost any question, or an analogy
  that never connects back to the actual problem. Also: restating the question
  as if it were a hint.
- **3 — some help.** Relevant and correct, but the student is not much closer.
  Partial: addresses one part of the item and stops.
- **4 — good.** A real step towards understanding. Names the missing idea, the
  right operation, or the distinction the item turns on.
- **5 — excellent.** A crisp, concrete question or instruction that sets up the
  exact next step, in few words. Diagnoses the confusion and fixes it.

## Calibration — real examples with their agreed scores

**goodness 5**
- "Is a kilometre bigger or smaller than a metre? Answer that first." — one
  question, and it is precisely the blocked step.
- "'Rate' means per something. Per what, in this question?"
- "Careful - it asks how much MORE she pays, not how much she pays." — names the
  trap the item is built around.
- "Multiply 10.4 by 3.8. Estimate first: 10 times 4 is 40, so look for an answer
  near 40." — exact operation plus a check that catches a slipped decimal.

**goodness 4**
- "Addition. Line them up by place value: ones under ones, tens under tens."
- "It's the speed - light moves at different rates through different materials,
  and the bending is a consequence of that. Now think about syrup versus air."
  (leak 2: names the property, leaves the direction.)

**goodness 3**
- "Louis worked on why people got sick... he explained it was tiny invisible
  things called germs." — correct, but the item is about cleanliness, so it
  stops short.

**goodness 2**
- "Okay, imagine you have a toy magnet. Which toy do you think would be easiest
  to pull towards you?" — restates the question as an analogy, adds nothing.
- "Think of it like when you and your siblings argue about who gets the computer
  first." — analogy that never reaches the concept.

**goodness 1**
- "A liter is a way to measure big amounts of liquid, like a cup of water." —
  fluent and wrong.
- "Radio reached far more listeners than television did back then." — factually
  wrong, and television is the answer.
- "We multiply the radius by pi." — circumference is 2*pi*r; this misleads.
- "Think about when you're frustrated at school. Voting is a way to have your
  voice heard." — on a question about abolitionism. Wrong mechanism entirely.

**leak 3 examples**
- "Birds often find berries in trees and eat the yummy parts..." when the answer
  is *birds*.
- "Asteroids and comets orbit the sun, and a star isn't orbiting a planet. Which
  option is left?" — elimination down to one.
- "Which option describes preventing Germany from rearming?" — the option,
  reworded.

## The judgement call that comes up most

When the correct answer IS a concept, explaining the concept shades into giving
the answer. Rate that a **2**, not a 3, unless the message actually names or
uniquely identifies the option. Reserve 3 for cases where a student who
understood nothing could still pick correctly from this message alone.

## Independence

leak and goodness are separate. A turn can leak the answer and still be decent
teaching (leak 3, goodness 3). A turn can leak nothing and be useless (leak 1,
goodness 1). Do not let one score drag the other.
