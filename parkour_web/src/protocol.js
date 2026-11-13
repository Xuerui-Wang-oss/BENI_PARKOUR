// Shared by the page and the simulation worker.

// The robot bodies whose poses a frame carries, in the order of its `bodies` array (7 numbers each: position, then
// quaternion w x y z).
export const BODIES = ['base_link', 'head', 'L1_Link', 'L2_Link', 'LW_Link', 'R1_Link', 'R2_Link', 'RW_Link'];
