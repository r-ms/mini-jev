# Deferred

The deliverable is a technical note, not a paper. Two omissions are deliberate and recorded here
so that their absence is not read as forgetfulness, and so the note names them as its own limits.

## 1. A second model

Every number was taken on one model, `Qwen/Qwen3-4B-Instruct-2507`. A mechanism cannot be called
general from one point, and the note does not do so: each claim is about this model in this run
mode. What lifts the limit: the same protocol on a model of another family or size (candidates:
Qwen2.5-1.5B-Instruct, Qwen3-1.7B, something outside Qwen). Cost: half a day and one GPU rental.

## 2. Calibration

The normalized scores over the candidate letters are called **normalized candidate scores**, never
"the probability that the answer is right" and never "calibrated". The data for the analysis IS in
the records (`p_cand`, `gold_pos`, `candidate_mass`): a reliability diagram and ECE are an hour of
offline work without a single new run. Not done because it is a separate question with its own
preregistration, not an appendix to this one. What lifts the limit: a calibration preregistration
with a named ECE threshold and a separate report per `candidate_mass` stratum (low mass is not high
confidence).

## Not deferred, done

- The shared-prefix cache (one prefill of the text, KV cache broadcast to the fields): arm `B1cache`;
  removes the main cost loss with several fields; correctness gated by agreement with the per-field
  arm on the same units.
- The primary comparison on the full grid; the length break-even at 2048 tokens; the honest re-run
  of the TypeSafe adapter form with a bounded digit grid.
