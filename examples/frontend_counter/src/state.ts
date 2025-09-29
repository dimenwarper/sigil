export interface CounterState {
  count: number;
  history: number[];
}

export type CounterAction =
  | { type: "increment"; amount: number }
  | { type: "reset" };

export function updateCounter(state: CounterState, action: CounterAction): CounterState {
  const history = state.history.slice();
  let nextCount = state.count;

  if (action.type === "increment") {
    for (let i = 0; i < action.amount; i += 1) {
      nextCount = nextCount + 1;
    }
  } else if (action.type === "reset") {
    nextCount = 0;
  }

  history.push(nextCount);
  return { count: nextCount, history };
}

export function initialState(): CounterState {
  return { count: 0, history: [] };
}
