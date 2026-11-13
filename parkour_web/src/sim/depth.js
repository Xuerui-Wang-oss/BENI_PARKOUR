// The robot's front depth camera, by ray casting against the terrain (geom group 0), as mjlab renders it for
// training: 60 x 80 pixels, pinhole camera with the vertical field of view of the MJCF camera, planar depth in
// metres (distance along the optical axis), 0 where a ray hits nothing. The robot's own geoms are not in the image.

export const DEPTH_ROWS = 60;
export const DEPTH_COLUMNS = 80;
const TERRAIN_GROUP = [1, 0, 0, 0, 0, 0];

export class DepthCamera {
  /** @param robot a `Robot`; its `depth_front` camera is used. */
  constructor(robot) {
    this.robot = robot;
    const { mj, model, camera } = robot;
    const count = DEPTH_ROWS * DEPTH_COLUMNS;
    const halfHeight = Math.tan((model.cam_fovy[camera] * Math.PI) / 360);
    const halfWidth = (halfHeight * DEPTH_COLUMNS) / DEPTH_ROWS;
    // Unit ray directions in the camera frame (x right, y up, looking along -z), row 0 at the top of the image,
    // and the cosine of each ray's angle to the optical axis.
    this.rays = new Float64Array(3 * count);
    this.cosine = new Float64Array(count);
    for (let row = 0; row < DEPTH_ROWS; row++) {
      for (let column = 0; column < DEPTH_COLUMNS; column++) {
        const x = -halfWidth + (2 * halfWidth * (column + 0.5)) / DEPTH_COLUMNS;
        const y = halfHeight - (2 * halfHeight * (row + 0.5)) / DEPTH_ROWS;
        const length = Math.hypot(x, y, 1);
        const i = row * DEPTH_COLUMNS + column;
        this.rays.set([x / length, y / length, -1 / length], 3 * i);
        this.cosine[i] = 1 / length;
      }
    }
    this.directions = new Array(3 * count).fill(0); // World frame; a plain array, as the binding takes it.
    this.geomIds = new mj.IntBuffer(count);
    this.distances = new mj.DoubleBuffer(count);
    this.image = new Float32Array(count);
  }

  /** Render the current state (call after `Robot.forward`); returns the image, row by row, reused between calls. */
  render() {
    const { mj, model, data, camera } = this.robot;
    const position = Array.from(data.cam_xpos.subarray(3 * camera, 3 * camera + 3));
    const r = data.cam_xmat.subarray(9 * camera, 9 * camera + 9); // Camera axes as columns, row-major.
    const { rays, directions, image, cosine } = this;
    for (let i = 0; i < rays.length; i += 3) {
      const x = rays[i];
      const y = rays[i + 1];
      const z = rays[i + 2];
      directions[i] = r[0] * x + r[1] * y + r[2] * z;
      directions[i + 1] = r[3] * x + r[4] * y + r[5] * z;
      directions[i + 2] = r[6] * x + r[7] * y + r[8] * z;
    }
    mj.mj_multiRay(model, data, position, directions, TERRAIN_GROUP, true, -1, this.geomIds, this.distances, null,
      image.length, 1e10);
    const distance = this.distances.GetView();
    for (let i = 0; i < image.length; i++) image[i] = distance[i] < 0 ? 0 : distance[i] * cosine[i];
    return image;
  }
}
