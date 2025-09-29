import React, { useState } from "react";
import { initialState, updateCounter } from "./state";

export function App(): JSX.Element {
  const [state, setState] = useState(initialState());

  return (
    <div style={{ fontFamily: "sans-serif", padding: "1rem" }}>
      <h1>Counter Demo</h1>
      <p>Count: {state.count}</p>
      <button
        type="button"
        onClick={() => setState(updateCounter(state, { type: "increment", amount: 1 }))}
      >
        Increment
      </button>
      <button type="button" onClick={() => setState(updateCounter(state, { type: "reset" }))}>
        Reset
      </button>
    </div>
  );
}

export default App;
