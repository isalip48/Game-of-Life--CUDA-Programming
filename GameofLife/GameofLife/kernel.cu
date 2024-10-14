// Include necessary libraries
#include <SDL.h>              // SDL for graphics
#include <cuda_runtime.h>     // CUDA runtime
#include <iostream>           // For standard I/O operations
#include <random>             // For random number generation
#include "device_launch_parameters.h"  // For CUDA kernel launch parameters

// Define constants for the game grid and window
const int GRID_WIDTH = 100;    // Width of the game grid
const int GRID_HEIGHT = 80;   // Height of the game grid
const int CELL_SIZE = 10;     // Size of each cell in pixels (smaller size now)
const int WINDOW_WIDTH = GRID_WIDTH * CELL_SIZE;   // Window width
const int WINDOW_HEIGHT = GRID_HEIGHT * CELL_SIZE; // Window height
const int BLOCK_SIZE = 2;    // CUDA block size
const int FRAME_DELAY = 9;  // Delay between frames in milliseconds to control the rendering speed
const int UPDATE_FREQUENCY = 6;  // Update grid every 5 frames

// CUDA kernel for updating the Game of Life grid
__global__ void updateGrid(unsigned char* d_grid, unsigned char* d_newGrid, int width, int height) { //This is the CUDA kernel function that runs on the GPU. It updates the state of the game grid.
    int x = blockIdx.x * blockDim.x + threadIdx.x; //Computes the x-coordinate for the cell that this thread is responsible for. Each thread in a CUDA block computes one cell.
    int y = blockIdx.y * blockDim.y + threadIdx.y; //Computes the y-coordinate similarly.

    if (x < width && y < height) {
        int idx = y * width + x; //Converts the (x, y) coordinates to a 1D index, which is used to access the 1D array representing the 2D grid.
        int count = 0;

        // Check all 8 neighboring cells
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0) continue;
                int nx = (x + dx + width) % width;
                int ny = (y + dy + height) % height;
                count += d_grid[ny * width + nx];
            } //These loops check the eight neighbors around the current cell (ignoring the cell itself).
        }

        // Apply Game of Life rules
        d_newGrid[idx] = (count == 3 || (count == 2 && d_grid[idx])) ? 1 : 0; //If the cell is alive and has exactly 2 or 3 neighbors, it stays alive. If the cell is dead and has exactly 3 neighbors, it becomes alive. Otherwise, it stays dead.


    }
}

// Initialize the grid with a random pattern
void initializeGrid(unsigned char* grid, int width, int height) {
    std::random_device rd;  // Obtain a random number from hardware
    std::mt19937 gen(rd()); // Seed the generator. 
    std::uniform_int_distribution<> dis(0, 1);  // Define the range

    // Set each cell to either 0 or 1 randomly
    for (int i = 0; i < width * height; i++) {
        grid[i] = dis(gen); // iterates over the entire grid and assigns each cell a random value (0 or 1).

    }
}

int main(int argc, char* argv[]) {
    // Initialize SDL
    if (SDL_Init(SDL_INIT_VIDEO) < 0) { //Initializes SDL for video rendering.
        std::cerr << "SDL could not initialize! SDL_Error: " << SDL_GetError() << std::endl;
        return -1;
    }

    // Create window
    SDL_Window* window = SDL_CreateWindow("Conway's Game of Life", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, WINDOW_WIDTH, WINDOW_HEIGHT, SDL_WINDOW_SHOWN); //Creates the SDL window with the dimensions
    if (window == nullptr) {
        std::cerr << "Window could not be created! SDL_Error: " << SDL_GetError() << std::endl;
        return -1;
    }

    // Create renderer
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED); //Creates the SDL renderer, which is used to draw the grid.
    if (renderer == nullptr) {
        std::cerr << "Renderer could not be created! SDL_Error: " << SDL_GetError() << std::endl;
        return -1;
    }

    // Allocate memory for the grid on the host
    unsigned char* h_grid = new unsigned char[GRID_WIDTH * GRID_HEIGHT]; //Allocates memory for the grid on the host (CPU).
    initializeGrid(h_grid, GRID_WIDTH, GRID_HEIGHT); //Initializes the grid with a random pattern.

    // Allocate memory for the grid on the device (GPU)
    unsigned char* d_grid;
    unsigned char* d_newGrid;
    cudaMalloc(&d_grid, GRID_WIDTH * GRID_HEIGHT * sizeof(unsigned char)); //Allocates memory on the GPU for both the current grid
    cudaMalloc(&d_newGrid, GRID_WIDTH * GRID_HEIGHT * sizeof(unsigned char)); //Copies the grid data from the host (CPU) to the GPU.
    // Copy the initial grid from host to device
    cudaMemcpy(d_grid, h_grid, GRID_WIDTH * GRID_HEIGHT * sizeof(unsigned char), cudaMemcpyHostToDevice);

    // Set up CUDA execution configuration
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((GRID_WIDTH + BLOCK_SIZE - 1) / BLOCK_SIZE, (GRID_HEIGHT + BLOCK_SIZE - 1) / BLOCK_SIZE);

    bool quit = false;  // Flag to exit the main loop
    SDL_Event e;        // SDL event handler
    int frameCount = 0; // Counter for frames

    // Main loop that continues until the user closes the window.
    while (!quit) {
        // Handle events on queue
        while (SDL_PollEvent(&e) != 0) { //Handles SDL events, like closing the window.
            if (e.type == SDL_QUIT) {
                quit = true;  // User requests quit
            }
        }

        // Update the grid every UPDATE_FREQUENCY frames
        if (frameCount % UPDATE_FREQUENCY == 0) { //Checks if it's time to update the grid.
            // Launch CUDA kernel
            updateGrid << <gridSize, blockSize >> > (d_grid, d_newGrid, GRID_WIDTH, GRID_HEIGHT);
            // Swap grid pointers
            unsigned char* temp = d_grid;
            d_grid = d_newGrid;
            d_newGrid = temp;
            // Copy updated grid from device to host
            cudaMemcpy(h_grid, d_grid, GRID_WIDTH * GRID_HEIGHT * sizeof(unsigned char), cudaMemcpyDeviceToHost);
        }

        // Clear screen
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);  // Set color to black
        SDL_RenderClear(renderer);

        // Render grid
        SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255);  // Set color to white
        for (int y = 0; y < GRID_HEIGHT; y++) {
            for (int x = 0; x < GRID_WIDTH; x++) {
                if (h_grid[y * GRID_WIDTH + x]) {  // If cell is alive
                    // Create a rectangle for the cell
                    SDL_Rect cellRect = { x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE };
                    SDL_RenderFillRect(renderer, &cellRect);  // Draw the cell
                }
            }
        }

        // Update screen
        SDL_RenderPresent(renderer);
        SDL_Delay(FRAME_DELAY);  // Add delay to control frame rate
        frameCount++;  // Increment frame counter
    }

    // Frees the GPU memory used for the grids.
    cudaFree(d_grid);
    cudaFree(d_newGrid);
    // Frees the memory used by the host grid.
    delete[] h_grid;

    // Destroy renderer
    SDL_DestroyRenderer(renderer);
    // Destroy window
    SDL_DestroyWindow(window);
    // Quit SDL subsystems
    SDL_Quit();

    return 0;  // Exit program
}
