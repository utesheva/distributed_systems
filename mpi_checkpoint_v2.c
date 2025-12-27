#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include <mpi-ext.h>
#include <signal.h>
#include <setjmp.h>
#include <string.h>
#include <unistd.h>

#define Max(a, b) ((a) > (b) ? (a) : (b))
#define N (4096 + 2)

double maxeps = 0.1e-7;
int itmax = 100;

double **A = NULL, **B = NULL;
double eps;
int rank_world, size_world;
MPI_Comm main_comm;
MPI_Errhandler errh;

int iteration_to_crash = -1, rank_to_crash = -1, it = 1;
static int has_died = 0, after_recovery = 0;
static jmp_buf recovery_jump;

static char program_path[512];
static char checkpoint_path[512];

int local_start_i = 0, local_end_i = 0, local_height = 0;

void calculate_distribution();
void allocate_local_arrays();
void free_local_arrays();
void init();
void relax();
void verify();
void save_checkpoint(int iteration);
int load_checkpoint(int *iteration);
void error_handler(MPI_Comm *comm, int *error_code, ... );
void spawned_process_main(MPI_Comm parent_comm);

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    
    char cwd[256];
    getcwd(cwd, sizeof(cwd));
    snprintf(program_path, sizeof(program_path), "%s/%s", cwd, argv[0]);
    snprintf(checkpoint_path, sizeof(checkpoint_path), "%s/checkpoint.dat", cwd);
    
    MPI_Comm parent_comm;
    MPI_Comm_get_parent(&parent_comm);
    
    if (parent_comm == MPI_COMM_NULL) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_world);
        MPI_Comm_size(MPI_COMM_WORLD, &size_world);
        
        if (argc == 3) {
            iteration_to_crash = atoi(argv[1]);
            rank_to_crash = atoi(argv[2]);
        }
        
        main_comm = MPI_COMM_WORLD;
        MPI_Comm_create_errhandler(error_handler, &errh);
        MPI_Comm_set_errhandler(main_comm, errh);
        
        if (rank_world == 0) {
            printf("=== Scenario B:  Dynamic Process Spawning ===\n");
            printf("Processes: %d\n\n", size_world);
        }
        
        setjmp(recovery_jump);
        
        if (! after_recovery) {
            calculate_distribution();
            allocate_local_arrays();
            
            int iteration_start;
            if (!load_checkpoint(&iteration_start)) {
                init();
                iteration_start = 1;
            }
            it = iteration_start;
        }
        
        while (it <= itmax) {
            if (rank_world == rank_to_crash && it == iteration_to_crash && ! has_died) {
                printf("Process %d failure at iteration %d\n", rank_world, it);
                fflush(stdout);
                raise(SIGKILL);
            }
            
            relax();
            
            if (rank_world == 0 && it % 10 == 0) {
                printf("Iteration %4d, eps = %.10e\n", it, eps);
            }
            
            if (it % 10 == 0) save_checkpoint(it);
            if (eps < maxeps) break;
            
            MPI_Barrier(main_comm);
            it++;
        }
        
        verify();
        free_local_arrays();
        
    } else {
        spawned_process_main(parent_comm);
        MPI_Comm_free(&parent_comm);
    }
    
    MPI_Finalize();
    return 0;
}

void spawned_process_main(MPI_Comm parent_comm)
{
    MPI_Comm merged_comm;
    MPI_Intercomm_merge(parent_comm, 1, &merged_comm);
    main_comm = merged_comm;
    
    MPI_Comm_rank(main_comm, &rank_world);
    MPI_Comm_size(main_comm, &size_world);
    MPI_Comm_create_errhandler(error_handler, &errh);
    MPI_Comm_set_errhandler(main_comm, errh);
    
    int path_len;
    MPI_Bcast(&path_len, 1, MPI_INT, 0, main_comm);
    MPI_Bcast(checkpoint_path, path_len, MPI_CHAR, 0, main_comm);
    MPI_Bcast(&path_len, 1, MPI_INT, 0, main_comm);
    MPI_Bcast(program_path, path_len, MPI_CHAR, 0, main_comm);
    MPI_Bcast(&it, 1, MPI_INT, 0, main_comm);
    
    calculate_distribution();
    allocate_local_arrays();
    
    int iteration_start;
    if (! load_checkpoint(&iteration_start)) {
        init();
        iteration_start = 1;
    }
    it = iteration_start;
    
    while (it <= itmax) {
        relax();
        if (it % 10 == 0) save_checkpoint(it);
        if (eps < maxeps) break;
        MPI_Barrier(main_comm);
        it++;
    }
    
    verify();
    free_local_arrays();
    MPI_Comm_free(&merged_comm);
}

void calculate_distribution()
{
    MPI_Comm_rank(main_comm, &rank_world);
    int rank_size;
    MPI_Comm_size(main_comm, &rank_size);
    local_start_i = rank_world * N / rank_size;
    local_end_i = (rank_world + 1) * N / rank_size;
    if (rank_world == rank_size - 1) local_end_i = N;
    local_height = local_end_i - local_start_i + 2;
}

void allocate_local_arrays()
{
    A = malloc(local_height * sizeof(double*));
    B = malloc(local_height * sizeof(double*));
    for (int i = 0; i < local_height; i++) {
        A[i] = calloc(N, sizeof(double));
        B[i] = calloc(N, sizeof(double));
    }
}

void free_local_arrays()
{
    if (A) {
        for (int i = 0; i < local_height; i++) free(A[i]);
        free(A);
        A = NULL;
    }
    if (B) {
        for (int i = 0; i < local_height; i++) free(B[i]);
        free(B);
        B = NULL;
    }
}

void init()
{
    for (int local_i = 1; local_i < local_height - 1; local_i++) {
        int global_i = local_start_i + local_i - 1;
        for (int j = 0; j < N; j++) {
            A[local_i][j] = (global_i == 0 || global_i == N-1 || j == 0 || j == N-1) 
                            ? 0.0 : (1.0 + global_i + j);
        }
    }
}

void relax()
{
    int rank_size;
    MPI_Comm_size(main_comm, &rank_size);
    MPI_Request req[4];
    int req_cnt = 0;
    
    if (rank_world < rank_size - 1) {
        MPI_Isend(A[local_height-2], N, MPI_DOUBLE, rank_world+1, 0, main_comm, &req[req_cnt++]);
        MPI_Irecv(A[local_height-1], N, MPI_DOUBLE, rank_world+1, 0, main_comm, &req[req_cnt++]);
    }
    if (rank_world > 0) {
        MPI_Isend(A[1], N, MPI_DOUBLE, rank_world-1, 0, main_comm, &req[req_cnt++]);
        MPI_Irecv(A[0], N, MPI_DOUBLE, rank_world-1, 0, main_comm, &req[req_cnt++]);
    }
    
    MPI_Waitall(req_cnt, req, MPI_STATUSES_IGNORE);
    
    int start = (rank_world == 0) ? 2 : 1;
    int end = (rank_world == rank_size-1) ? local_height-2 : local_height-1;
    
    for (int i = start; i < end; i++) {
        for (int j = 1; j < N-1; j++) {
            B[i][j] = 0.25 * (A[i-1][j] + A[i+1][j] + A[i][j-1] + A[i][j+1]);
        }
    }
    
    double local_eps = 0.0;
    for (int i = start; i < end; i++) {
        for (int j = 1; j < N-1; j++) {
            local_eps = Max(local_eps, fabs(A[i][j] - B[i][j]));
            A[i][j] = B[i][j];
        }
    }
    
    MPI_Allreduce(&local_eps, &eps, 1, MPI_DOUBLE, MPI_MAX, main_comm);
}

void save_checkpoint(int iteration)
{
    MPI_File fh;
    if (MPI_File_open(main_comm, checkpoint_path, MPI_MODE_WRONLY | MPI_MODE_CREATE, 
                      MPI_INFO_NULL, &fh) != MPI_SUCCESS) return;
    
    int rank_size;
    MPI_Comm_size(main_comm, &rank_size);
    
    if (rank_world == 0) {
        int meta[3] = {iteration, rank_size, N};
        MPI_File_write_at(fh, 0, meta, 3, MPI_INT, MPI_STATUS_IGNORE);
    }
    MPI_Barrier(main_comm);
    
    MPI_Offset offset = 3 * sizeof(int);
    for (int r = 0; r < rank_world; r++) {
        int h = ((r+1)*N/rank_size) - (r*N/rank_size);
        if (r == rank_size-1) h = N - (r*N/rank_size);
        offset += h * N * sizeof(double);
    }
    
    int rows = local_end_i - local_start_i;
    for (int i = 1; i <= rows; i++) {
        MPI_File_write_at(fh, offset + (i-1)*N*sizeof(double), A[i], N, MPI_DOUBLE, MPI_STATUS_IGNORE);
    }
    
    MPI_File_close(&fh);
}

int load_checkpoint(int *iteration)
{
    if (rank_world == 0) {
        FILE *f = fopen(checkpoint_path, "r");
        *iteration = f ? 1 : 0;
        if (f) fclose(f);
    }
    MPI_Bcast(iteration, 1, MPI_INT, 0, main_comm);
    if (! *iteration) return 0;
    
    MPI_File fh;
    if (MPI_File_open(main_comm, checkpoint_path, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh) != MPI_SUCCESS) 
        return 0;
    
    int meta[3];
    if (rank_world == 0) MPI_File_read_at(fh, 0, meta, 3, MPI_INT, MPI_STATUS_IGNORE);
    MPI_Bcast(meta, 3, MPI_INT, 0, main_comm);
    
    int rank_size;
    MPI_Comm_size(main_comm, &rank_size);
    if (meta[1] != rank_size || meta[2] != N) {
        MPI_File_close(&fh);
        return 0;
    }
    
    MPI_Offset offset = 3 * sizeof(int);
    for (int r = 0; r < rank_world; r++) {
        int h = ((r+1)*N/rank_size) - (r*N/rank_size);
        if (r == rank_size-1) h = N - (r*N/rank_size);
        offset += h * N * sizeof(double);
    }
    
    int rows = local_end_i - local_start_i;
    for (int i = 1; i <= rows; i++) {
        MPI_File_read_at(fh, offset + (i-1)*N*sizeof(double), A[i], N, MPI_DOUBLE, MPI_STATUS_IGNORE);
    }
    
    MPI_File_close(&fh);
    
    // Share
    MPI_Request req[4];
    int req_cnt = 0;
    if (rank_world < rank_size-1) {
        MPI_Isend(A[local_height-2], N, MPI_DOUBLE, rank_world+1, 0, main_comm, &req[req_cnt++]);
        MPI_Irecv(A[local_height-1], N, MPI_DOUBLE, rank_world+1, 0, main_comm, &req[req_cnt++]);
    }
    if (rank_world > 0) {
        MPI_Isend(A[1], N, MPI_DOUBLE, rank_world-1, 0, main_comm, &req[req_cnt++]);
        MPI_Irecv(A[0], N, MPI_DOUBLE, rank_world-1, 0, main_comm, &req[req_cnt++]);
    }
    MPI_Waitall(req_cnt, req, MPI_STATUSES_IGNORE);
    
    *iteration = meta[0] + 1;
    return 1;
}

void verify()
{
    double local_sum = 0.0;
    for (int i = 1; i < local_height-1; i++) {
        int gi = local_start_i + i - 1;
        for (int j = 0; j < N; j++) {
            local_sum += A[i][j] * (gi+1) * (j+1) / (N*N);
        }
    }
    double sum;
    MPI_Reduce(&local_sum, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, main_comm);
    if (rank_world == 0) printf("S = %f\n", sum);
}

void error_handler(MPI_Comm *pcomm, int *error_code, ...)
{
    if (has_died) return;
    has_died = 1;
    
    MPIX_Comm_revoke(*pcomm);
    MPI_Comm shrinked;
    MPIX_Comm_shrink(*pcomm, &shrinked);
    
    int old_sz, new_sz;
    MPI_Comm_size(*pcomm, &old_sz);
    MPI_Comm_size(shrinked, &new_sz);
    int failed = old_sz - new_sz;
    
    if (rank_world == 0) {
        printf("\n=== FAILURE:  %d failed, spawning replacements ===\n", failed);
    }
    
    // Spawn
    MPI_Info info;
    MPI_Info_create(&info);
    char cwd[256];
    if (getcwd(cwd, sizeof(cwd))) MPI_Info_set(info, "wdir", cwd);
    
    MPI_Comm intercomm;
    char *argv[1] = {NULL};
    int *errs = malloc(failed * sizeof(int));
    
    int rc = MPI_Comm_spawn(program_path, argv, failed, info, 0, shrinked, &intercomm, errs);
    MPI_Info_free(&info);
    free(errs);
    
    if (rc != MPI_SUCCESS) {
        printf("Error with spawning\n");
        exit(1);
    }
    
    if (rank_world == 0) printf("=== Spawned %d, merging ===\n", failed);
    
    sleep(1);
    MPI_Comm new_comm;
    MPI_Intercomm_merge(intercomm, 0, &new_comm);
    
    int len = strlen(checkpoint_path) + 1;
    MPI_Bcast(&len, 1, MPI_INT, 0, new_comm);
    MPI_Bcast(checkpoint_path, len, MPI_CHAR, 0, new_comm);
    len = strlen(program_path) + 1;
    MPI_Bcast(&len, 1, MPI_INT, 0, new_comm);
    MPI_Bcast(program_path, len, MPI_CHAR, 0, new_comm);
    MPI_Bcast(&it, 1, MPI_INT, 0, new_comm);
    
    free_local_arrays();
    main_comm = new_comm;
    MPI_Comm_set_errhandler(main_comm, errh);
    MPI_Comm_rank(main_comm, &rank_world);
    MPI_Comm_size(main_comm, &size_world);
    calculate_distribution();
    allocate_local_arrays();
    
    int iter;
    if (!load_checkpoint(&iter)) { init(); iter = 1; }
    it = iter;
    
    if (rank_world == 0) printf("=== RECOVERED:  size=%d, iteration=%d ===\n\n", size_world, it);
    
    MPI_Comm_free(&intercomm);
    MPI_Comm_free(&shrinked);
    after_recovery = 1;
    longjmp(recovery_jump, 1);
}
