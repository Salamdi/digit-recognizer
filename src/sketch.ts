import { multiply, exp, map } from 'mathjs';
import p5 from 'p5';
import model from './model.json';

const sketch = (p: p5) => {
  let canvas: p5.Renderer;

  p.setup = () => {
    // Create a 280x280 canvas
    canvas = p.createCanvas(280, 280);
    // Attach canvas to the container in index.html
    const container = document.getElementById('canvas-container');
    if (container) {
      container.appendChild(canvas.elt);
    }
    // Set background and drawing settings
    p.background(255);
    p.stroke(0);
    p.strokeWeight(16);

    // Bind button events (using standard DOM)
    const clearBtn = document.getElementById('clearBtn');
    if (clearBtn) {
      clearBtn.addEventListener('click', clearCanvas);
    }
    const convertBtn = document.getElementById('convertBtn');
    if (convertBtn) {
      convertBtn.addEventListener('click', convertDrawing);
    }
  };

  p.draw = () => {
    // Draw lines while the mouse is pressed and within canvas bounds
    if (
      p.mouseIsPressed &&
      p.mouseX >= 0 &&
      p.mouseX < p.width &&
      p.mouseY >= 0 &&
      p.mouseY < p.height
    ) {
      p.line(p.pmouseX, p.pmouseY, p.mouseX, p.mouseY);
    }
  };

  const clearCanvas = (): void => {
    p.background(255);
  };

  const predict = (input: number[]) => {
    let output = input;

    for (let layer of model) {
      const z = multiply(layer, [1, ...output]);
      output = map(z, (o) => exp(o) / (1 + exp(o)));
    }

    let max = -Infinity;
    let maxi = -1;

    for (let i = 0; i < output.length; i++) {
      if (output[i] > max) {
        max = output[i];
        maxi = i;
      }
    }

    return maxi;
  };

  const convertDrawing = (): void => {
    // Update the pixel array from the canvas
    p.loadPixels();
    const cellSize = 10 * p.pixelDensity(); // 280 / 28 = 10
    const grid: number[][] = [];

    // Iterate over each cell in the 28x28 grid
    for (let row = 0; row < 28; row++) {
      const gridRow: number[] = [];
      for (let col = 0; col < 28; col++) {
        let cellHasDrawing = false;
        // Check each pixel in the 10x10 block
        for (let y = row * cellSize; y < (row + 1) * cellSize; y++) {
          for (let x = col * cellSize; x < (col + 1) * cellSize; x++) {
            const index = 4 * (y * p.width * p.pixelDensity() + x);
            // The canvas background is white (255); any pixel drawn (black) will be lower.
            if (p.pixels[index] < 150) {
              cellHasDrawing = true;
              break;
            }
          }
          if (cellHasDrawing) break;
        }
        gridRow.push(cellHasDrawing ? 1 : 0);
      }
      grid.push(gridRow);
    }

    // Log and display the grid
    const result = predict(grid.flat());
    const matrixPre = document.getElementById('matrix');
    if (matrixPre) {
      matrixPre.textContent = result.toString(10);
    }
  };
};

// Create a new p5 instance with our sketch
new p5(sketch);
