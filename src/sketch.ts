import { multiply, exp, map } from 'mathjs';
import model from './model.json';

const WIDTH = 280;
const HEIGHT = 280;
const GRID_SIZE = 28;
const CELL = WIDTH / GRID_SIZE; // 10px per cell

const container = document.getElementById('canvas-container')!;
const canvas = document.createElement('canvas');
const ctx = canvas.getContext('2d')!;

const dpr = window.devicePixelRatio || 1;
canvas.width = WIDTH * dpr;
canvas.height = HEIGHT * dpr;
canvas.style.width = `${WIDTH}px`;
canvas.style.height = `${HEIGHT}px`;
ctx.scale(dpr, dpr);

container.appendChild(canvas);

ctx.fillStyle = 'white';
ctx.fillRect(0, 0, WIDTH, HEIGHT);
ctx.strokeStyle = 'black';
ctx.lineWidth = 16;
ctx.lineCap = 'round';
ctx.lineJoin = 'round';

let drawing = false;
let prevX = 0;
let prevY = 0;

function getPos(e: MouseEvent) {
  const rect = canvas.getBoundingClientRect();
  return { x: e.clientX - rect.left, y: e.clientY - rect.top };
}

canvas.addEventListener('mousedown', (e) => {
  drawing = true;
  const pos = getPos(e);
  prevX = pos.x;
  prevY = pos.y;
});

canvas.addEventListener('mousemove', (e) => {
  if (!drawing) return;
  const pos = getPos(e);
  ctx.beginPath();
  ctx.moveTo(prevX, prevY);
  ctx.lineTo(pos.x, pos.y);
  ctx.stroke();
  prevX = pos.x;
  prevY = pos.y;
});

canvas.addEventListener('mouseup', () => { drawing = false; });
canvas.addEventListener('mouseleave', () => { drawing = false; });

const clearCanvas = () => {
  ctx.fillStyle = 'white';
  ctx.fillRect(0, 0, WIDTH, HEIGHT);
};

const predict = (input: number[]) => {
  let output = input;
  for (const layer of model) {
    const z = multiply(layer, [1, ...output]);
    output = map(z, (o) => exp(o) / (1 + exp(o)));
  }
  let max = -Infinity;
  let maxi = -1;
  for (let i = 0; i < output.length; i++) {
    if (output[i] > max) { max = output[i]; maxi = i; }
  }
  return maxi;
};

const convertDrawing = () => {
  const cellSize = CELL * dpr;
  const { data: pixels } = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const grid: number[] = [];

  for (let row = 0; row < GRID_SIZE; row++) {
    for (let col = 0; col < GRID_SIZE; col++) {
      let hit = false;
      outer: for (let y = row * cellSize; y < (row + 1) * cellSize; y++) {
        for (let x = col * cellSize; x < (col + 1) * cellSize; x++) {
          if (pixels[4 * (Math.floor(y) * canvas.width + Math.floor(x))] < 150) {
            hit = true;
            break outer;
          }
        }
      }
      grid.push(hit ? 1 : 0);
    }
  }

  const result = predict(grid);
  const el = document.getElementById('matrix');
  if (el) el.textContent = result.toString(10);
};

document.getElementById('clearBtn')?.addEventListener('click', clearCanvas);
document.getElementById('convertBtn')?.addEventListener('click', convertDrawing);
