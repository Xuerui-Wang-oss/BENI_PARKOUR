"""An LLM in place of the PPO-trained high-level parkour policy.

The LLM gets what the high-level policy gets, in a form it can read: the front camera image (depth, RGB or both)
and the 24 body-state numbers as named text. It returns what the high-level policy returns: forward speed, body
height and a jump trigger. Everything below that boundary (``parkour_rl.hierarchy``) is unchanged.

    scene.py     the play env on chosen lanes, with a camera for the LLM next to the PPO depth camera
    views.py     camera frames -> PNG images and a table of distances
    prompt.py    the system prompt and the per-decision observation text
    backends.py  who answers: Claude, or a scripted oracle that reads the simulator (for testing the plumbing)
    policy.py    LlmPolicy: calls the backend every few steps and turns its answers into high-level actions
    run.py       run episodes, log every prompt and reply, print the score
"""
