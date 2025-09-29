import { initialState, updateCounter } from "./state";

function checkCorrectness(): void {
  let state = initialState();
  state = updateCounter(state, { type: "increment", amount: 5 });
  state = updateCounter(state, { type: "increment", amount: 3 });
  state = updateCounter(state, { type: "reset" });
  state = updateCounter(state, { type: "increment", amount: 2 });

  const expectedHistory = [5, 8, 0, 2];
  const isCorrect =
    state.count === 2 &&
    state.history.length === expectedHistory.length &&
    state.history.every((value, index) => value === expectedHistory[index]);

  if (isCorrect) {
    console.log("correctness=true");
    process.exit(0);
  }

  console.log("correctness=false");
  process.exit(1);
}

function measureLatency(): void {
  let state = initialState();
  const start = Date.now();
  for (let i = 0; i < 5000; i += 1) {
    state = updateCounter(state, { type: "increment", amount: 1 });
    if (i % 50 === 0) {
      state = updateCounter(state, { type: "increment", amount: 2 });
    }
  }
  const end = Date.now();
  console.log(`latency_ms=${(end - start).toFixed(3)}`);
}

function main() {
  const mode = process.argv[2];
  if (mode === "correctness") {
    checkCorrectness();
    return;
  }
  if (mode === "latency") {
    measureLatency();
    return;
  }
  console.error("Usage: ts-node src/eval.ts [correctness|latency]");
  process.exit(2);
}

main();
