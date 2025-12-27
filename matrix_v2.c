#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

#define GRID_SIZE 8
#define MESSAGE_LENGTH 1000000   
#define CHUNK_COUNT 50          
#define CHUNK_SIZE (MESSAGE_LENGTH / CHUNK_COUNT)

typedef struct {
    int x;
    int y;
} Coord;

typedef enum {
    ROUTE_HORIZONTAL_FIRST = 0,
    ROUTE_VERTICAL_FIRST = 1
} Route;

int coord_to_rank(int x, int y) {
    return y * GRID_SIZE + x;
}

Coord rank_to_coord(int rank) {
    Coord c;
    c.x = rank % GRID_SIZE;
    c.y = rank / GRID_SIZE;
    return c;
}

int manhattan_distance(Coord from, Coord to) {
    return abs(to.x - from.x) + abs(to.y - from. y);
}


int is_on_route(Coord my_coord, Coord source, Coord target, Route route) {
    if (route == ROUTE_HORIZONTAL_FIRST) {
        if (my_coord.y == source.y && my_coord.x > source.x && my_coord.x <= target.x) {
            return 1;
        }
        if (my_coord.x == target.x && my_coord.y > source.y && my_coord.y < target.y) {
            return 2;
        }
    } else {
        if (my_coord.x == source.x && my_coord.y > source.y && my_coord. y <= target.y) {
            return 3;
        }
        if (my_coord.y == target. y && my_coord.x > source.x && my_coord. x < target.x) {
            return 4;
        }
    }
    return 0;
}

void get_neighbors(Coord my_coord, Coord source, Coord target, Route route, 
                   int route_position, int *prev_rank, int *next_rank) {
    if (route == ROUTE_HORIZONTAL_FIRST) {
        if (route_position == 1) {
            *prev_rank = coord_to_rank(my_coord.x - 1, my_coord.y);
            if (my_coord.x < target.x) {
                *next_rank = coord_to_rank(my_coord.x + 1, my_coord.y);
            } else {
                *next_rank = coord_to_rank(my_coord.x, my_coord.y + 1);
            }
        } else if (route_position == 2) {
            *prev_rank = coord_to_rank(my_coord.x, my_coord.y - 1);
            *next_rank = coord_to_rank(my_coord.x, my_coord.y + 1);
        }
    } else {
        if (route_position == 3) {
            *prev_rank = coord_to_rank(my_coord.x, my_coord.y - 1);
            if (my_coord.y < target.y) {
                *next_rank = coord_to_rank(my_coord.x, my_coord.y + 1);
            } else {
                *next_rank = coord_to_rank(my_coord.x + 1, my_coord.y);
            }
        } else if (route_position == 4) {
            *prev_rank = coord_to_rank(my_coord.x - 1, my_coord.y);
            *next_rank = coord_to_rank(my_coord.x + 1, my_coord.y);
        }
    }
}

int main(int argc, char** argv) {
    int rank, size;
    MPI_Status status;
    double start_time, end_time;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (size != GRID_SIZE * GRID_SIZE) {
        if (rank == 0) {
            printf("Error: Programm need %d processes (grid %dx%d)\n", 
                   GRID_SIZE * GRID_SIZE, GRID_SIZE, GRID_SIZE);
            printf("Run: mpirun -np %d ./matrix_v2\n", 
                   GRID_SIZE * GRID_SIZE);
        }
        MPI_Finalize();
        return 1;
    }
    
    Coord my_coord = rank_to_coord(rank);
    Coord source = {0, 0};
    Coord target = {7, 7};
    
    int source_rank = coord_to_rank(source.x, source.y);
    int target_rank = coord_to_rank(target. x, target.y);
    char* message = (char*)malloc(MESSAGE_LENGTH * sizeof(char));
    
    if (rank == source_rank) {
        for (int i = 0; i < MESSAGE_LENGTH; i++) {
            message[i] = 'A' + (i % 26);
        }
        
        printf("Parameters\n");
        printf("Source: node (%d, %d) [rank %d]\n", source.x, source.y, source_rank);
        printf("Destination: node (%d, %d) [rank %d]\n", target.x, target.y, target_rank);
        printf("Length: %d байт\n", MESSAGE_LENGTH);
        printf("Number of parts: %d\n", CHUNK_COUNT);
        printf("Size of one part: %d byte\n", CHUNK_SIZE);
        
        start_time = MPI_Wtime();
        
        MPI_Request* requests = (MPI_Request*)malloc(CHUNK_COUNT * sizeof(MPI_Request));
        
        for (int chunk = 0; chunk < CHUNK_COUNT; chunk++) {
            int offset = chunk * CHUNK_SIZE;
            int current_chunk_size = (chunk == CHUNK_COUNT - 1) 
                                     ? (MESSAGE_LENGTH - offset) 
                                     : CHUNK_SIZE;
            
            int next_rank;
            Route route = (chunk % 2 == 0) ? ROUTE_HORIZONTAL_FIRST : ROUTE_VERTICAL_FIRST;
            
            if (route == ROUTE_HORIZONTAL_FIRST) {
                next_rank = coord_to_rank(my_coord.x + 1, my_coord.y);
            } else {
                next_rank = coord_to_rank(my_coord.x, my_coord.y + 1);
            }
            
            MPI_Isend(message + offset, current_chunk_size, MPI_CHAR, 
                     next_rank, chunk, MPI_COMM_WORLD, &requests[chunk]);
        }
        
        MPI_Waitall(CHUNK_COUNT, requests, MPI_STATUSES_IGNORE);
        free(requests);
        
        end_time = MPI_Wtime();
        
    } else if (rank == target_rank) {
        start_time = MPI_Wtime();
        
        MPI_Request* requests = (MPI_Request*)malloc(CHUNK_COUNT * sizeof(MPI_Request));
        
        for (int chunk = 0; chunk < CHUNK_COUNT; chunk++) {
            int offset = chunk * CHUNK_SIZE;
            int current_chunk_size = (chunk == CHUNK_COUNT - 1) 
                                     ? (MESSAGE_LENGTH - offset) 
                                     :  CHUNK_SIZE;
            
            int prev_rank;
            Route route = (chunk % 2 == 0) ? ROUTE_HORIZONTAL_FIRST : ROUTE_VERTICAL_FIRST;
            
            if (route == ROUTE_HORIZONTAL_FIRST) {
                prev_rank = coord_to_rank(my_coord.x, my_coord.y - 1);
            } else {
                prev_rank = coord_to_rank(my_coord.x - 1, my_coord.y);
            }
            
            MPI_Irecv(message + offset, current_chunk_size, MPI_CHAR, 
                     prev_rank, chunk, MPI_COMM_WORLD, &requests[chunk]);
        }
        
        MPI_Waitall(CHUNK_COUNT, requests, MPI_STATUSES_IGNORE);
        free(requests);
        
        end_time = MPI_Wtime();
        
        int correct = 1;
        for (int i = 0; i < MESSAGE_LENGTH && correct; i++) {
            if (message[i] != ('A' + (i % 26))) {
                correct = 0;
            }
        }
        
        printf("\nResults\n");
        printf("Recieved: %s\n", correct ? "Yes" : "No");
        printf("Time: %.6f seconds (%.2f mks)\n", 
               end_time - start_time, (end_time - start_time) * 1e6);
        
    } else {
        int route1_pos = is_on_route(my_coord, source, target, ROUTE_HORIZONTAL_FIRST);
        int route2_pos = is_on_route(my_coord, source, target, ROUTE_VERTICAL_FIRST);
        
        if (route1_pos || route2_pos) {
            MPI_Request* recv_requests = (MPI_Request*)malloc(CHUNK_COUNT * sizeof(MPI_Request));
            MPI_Request* send_requests = (MPI_Request*)malloc(CHUNK_COUNT * sizeof(MPI_Request));
            int active_chunks = 0;
            
            for (int chunk = 0; chunk < CHUNK_COUNT; chunk++) {
                Route route = (chunk % 2 == 0) ? ROUTE_HORIZONTAL_FIRST : ROUTE_VERTICAL_FIRST;
                int route_pos = (route == ROUTE_HORIZONTAL_FIRST) ? route1_pos : route2_pos;
                
                if (route_pos == 0) {
                    recv_requests[chunk] = MPI_REQUEST_NULL;
                    send_requests[chunk] = MPI_REQUEST_NULL;
                    continue;
                }
                
                int offset = chunk * CHUNK_SIZE;
                int current_chunk_size = (chunk == CHUNK_COUNT - 1) 
                                         ? (MESSAGE_LENGTH - offset) 
                                         :  CHUNK_SIZE;
                
                int prev_rank, next_rank;
                get_neighbors(my_coord, source, target, route, route_pos, 
                            &prev_rank, &next_rank);
                
                MPI_Irecv(message + offset, current_chunk_size, MPI_CHAR, 
                         prev_rank, chunk, MPI_COMM_WORLD, &recv_requests[chunk]);
                active_chunks++;
            }
            
            for (int chunk = 0; chunk < CHUNK_COUNT; chunk++) {
                if (recv_requests[chunk] == MPI_REQUEST_NULL) continue;
                
                MPI_Wait(&recv_requests[chunk], MPI_STATUS_IGNORE);
                Route route = (chunk % 2 == 0) ? ROUTE_HORIZONTAL_FIRST : ROUTE_VERTICAL_FIRST;
                int route_pos = (route == ROUTE_HORIZONTAL_FIRST) ? route1_pos : route2_pos;
                int offset = chunk * CHUNK_SIZE;
                int current_chunk_size = (chunk == CHUNK_COUNT - 1) 
                                         ? (MESSAGE_LENGTH - offset) 
                                         :  CHUNK_SIZE;
                int prev_rank, next_rank;
                get_neighbors(my_coord, source, target, route, route_pos, 
                            &prev_rank, &next_rank);
                MPI_Isend(message + offset, current_chunk_size, MPI_CHAR, 
                         next_rank, chunk, MPI_COMM_WORLD, &send_requests[chunk]);
            }
            
            MPI_Waitall(CHUNK_COUNT, send_requests, MPI_STATUSES_IGNORE);
            free(recv_requests);
            free(send_requests);
        }
    }
    
    free(message);
    MPI_Finalize();
    return 0;
}
