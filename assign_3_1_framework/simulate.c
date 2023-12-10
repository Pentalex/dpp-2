/*
   File: simulate.c

   Description:
   This file contains an MPI parallel implementation of a wave equation
   simulation.

   Functions:
   - MYMPI_Bcast: Custom broadcast function using point-to-point communication.
   - simulate_range: Simulates a range of elements in the array for a given time
   step.
   - exchange_halos: Exchanges halo cells between neighboring processes.
   - simulate: Main simulation function utilizing MPI parallelization.

*/

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "simulate.h"

/*
   Custom MPI broadcast function that distributes data from the root process
   to all other processes in the communicator.
*/
int MYMPI_Bcast(void *buffer, int count, MPI_Datatype datatype, int root,
                MPI_Comm communicator) {
  int rank, size;
  MPI_Comm_rank(communicator, &rank);
  MPI_Comm_size(communicator, &size);

  void *temp_buffer = malloc(count * MPI_Type_size(datatype, &size));

  if (rank == root) {
    // If root, send data to neighbors
    MPI_Send(buffer, count, datatype, (rank + 1) % size, 0, communicator);
    MPI_Send(buffer, count, datatype, (rank - 1 + size) % size, 0,
             communicator);
  } else {
    // If non-root, receive data from the root
    MPI_Recv(temp_buffer, count, datatype, (rank - 1 + size) % size, 0,
             communicator, MPI_STATUS_IGNORE);

    // Forward the received data to the next process
    MPI_Send(temp_buffer, count, datatype, (rank + 1) % size, 0, communicator);
  }

  free(temp_buffer);

  return MPI_SUCCESS;
}

/*
    Simulates a specific range of elements in the array for a given time
   step.
*/
void simulate_range(const int start, const int end, const int i_max,
                    double *old_array, double *current_array,
                    double *next_array) {
  for (int i = start; i <= end; ++i) {
    if (i > 0 && i < i_max - 1) {
      next_array[i] = 2.0 * current_array[i] - old_array[i] +
                      0.01 * (current_array[i - 1] -
                              (2.0 * current_array[i] - current_array[i + 1]));
    }
  }
}

/*
    Exchanges halo cells between neighboring processes.
*/
void exchange_halos(double *local_array, const int local_size, int rank,
                    int size) {
  MPI_Status status;

  // Send and receive halo cells with neighboring processes
  if (rank > 0) {
    MPI_Send(&local_array[1], 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD);
    MPI_Recv(&local_array[0], 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD,
             &status);
  }

  if (rank < size - 1) {
    MPI_Send(&local_array[local_size - 2], 1, MPI_DOUBLE, rank + 1, 0,
             MPI_COMM_WORLD);
    MPI_Recv(&local_array[local_size - 1], 1, MPI_DOUBLE, rank + 1, 0,
             MPI_COMM_WORLD, &status);
  }
}

/*    Main simulation function utilizing    */
double *simulate(const int i_max, const int t_max, double *old_array,
                 double *current_array, double *next_array) {
  int rank, size;

  MPI_Init(NULL, NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int local_size = i_max / size;
  int start = rank * local_size;
  int end = start + local_size - 1;

  for (int t = 0; t < t_max; ++t) {
    simulate_range(start + 1, end - 1, i_max, old_array, current_array,
                   next_array);

    exchange_halos(current_array, local_size, rank, size);

    double *temp = old_array;
    old_array = current_array;
    current_array = next_array;
    next_array = temp;
  }

  MPI_Finalize();

  return current_array;
}
