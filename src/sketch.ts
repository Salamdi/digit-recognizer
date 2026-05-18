import * as ort from 'onnxruntime-web';

ort.env.wasm.numThreads = 1;

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
let inferenceTimer: ReturnType<typeof setInterval> | null = null;

const startInferenceLoop = () => {
  if (inferenceTimer) return;
  inferenceTimer = setInterval(convertDrawing, 100);
};

const stopInferenceLoop = () => {
  if (inferenceTimer) { clearInterval(inferenceTimer); inferenceTimer = null; }
};

function getPos(e: MouseEvent) {
  const rect = canvas.getBoundingClientRect();
  return { x: e.clientX - rect.left, y: e.clientY - rect.top };
}

canvas.addEventListener('mousedown', (e) => {
  drawing = true;
  const pos = getPos(e);
  prevX = pos.x;
  prevY = pos.y;
  startInferenceLoop();
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

canvas.addEventListener('mouseup', () => { if (drawing) { drawing = false; stopInferenceLoop(); convertDrawing(); } });
canvas.addEventListener('mouseleave', () => { if (drawing) { drawing = false; stopInferenceLoop(); convertDrawing(); } });

const clearCanvas = () => {
  ctx.fillStyle = 'white';
  ctx.fillRect(0, 0, WIDTH, HEIGHT);
};

const toTensor = (): ort.Tensor => {
  const cellSize = CELL * dpr;
  const { data: pixels } = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const values = new Float32Array(GRID_SIZE * GRID_SIZE);

  for (let row = 0; row < GRID_SIZE; row++) {
    for (let col = 0; col < GRID_SIZE; col++) {
      let sum = 0;
      const yStart = Math.floor(row * cellSize);
      const yEnd = Math.floor((row + 1) * cellSize);
      const xStart = Math.floor(col * cellSize);
      const xEnd = Math.floor((col + 1) * cellSize);
      for (let y = yStart; y < yEnd; y++) {
        for (let x = xStart; x < xEnd; x++) {
          sum += pixels[4 * (y * canvas.width + x)]; // R channel (grayscale)
        }
      }
      const count = (yEnd - yStart) * (xEnd - xStart);
      // Invert: white background (255) → 0.0, black drawing (0) → 1.0
      values[row * GRID_SIZE + col] = (255 - sum / count) / 255;
    }
  }

  // Z-score normalization: mean=0, std=1
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const variance = values.reduce((a, b) => a + (b - mean) ** 2, 0) / values.length;
  const std = Math.sqrt(variance);
  if (std > 0) {
    for (let i = 0; i < values.length; i++) {
      values[i] = (values[i] - mean) / std;
    }
  }

  return new ort.Tensor('float32', values, [1, 1, GRID_SIZE, GRID_SIZE]);
};

const sessionPromise = ort.InferenceSession.create('/model.onnx', {
  executionProviders: ['wasm'],
});

const renderChart = (probs: number[], predicted: number): void => {
  const chartEl = document.getElementById('chart');
  if (!chartEl) return;
  const BAR_MAX_HEIGHT = 120;
  chartEl.innerHTML = probs.map((p, i) => {
    const height = Math.round(p * BAR_MAX_HEIGHT);
    const winner = i === predicted;
    return `<div class="bar-item${winner ? ' bar-item--winner' : ''}">
      <span class="bar-score">${(p * 100).toFixed(1)}%</span>
      <div class="bar" style="height:${height}px"></div>
      <span class="bar-label">${i}</span>
    </div>`;
  }).join('');
};

const convertDrawing = async () => {
  try {
    const session = await sessionPromise;
    const tensor = toTensor();
    const feeds = { [session.inputNames[0]]: tensor };
    const results = await session.run(feeds);
    const output = results[session.outputNames[0]].data as Float32Array;

    let maxVal = -Infinity;
    let predicted = -1;
    for (let i = 0; i < output.length; i++) {
      if (output[i] > maxVal) { maxVal = output[i]; predicted = i; }
    }

    renderChart(Array.from(output), predicted);
  } catch (err) {
    console.error('Inference failed:', err);
  }
};

document.getElementById('clearBtn')?.addEventListener('click', clearCanvas);
