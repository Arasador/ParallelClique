/**
Sistemas Distribuidos e Paralelos
CudaClique.cu
Purpose: Calculates max clique on graphs

@author Alana Rasador Panizzi
@version 1.1 13/07/16
*/

#include <thrust/logical.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <stdio.h>
#include <cassert>
#include <ctime>

#define N 100
#define NUM_TREADS 30
#define K 10
#define NUM_ADJ (N - 1)*N / 2
#define GRAPH_DENSITY 0.3
 
typedef thrust::device_vector<float> D_vector_float;
typedef thrust::host_vector<float> H_vector_float;
typedef thrust::host_vector<int> H_vector_int;
typedef thrust::host_vector<bool> H_vector_bool;
typedef thrust::device_vector<bool> D_vector_bool;
 


/*
 * calculates the index in the adjacencies matrix (upper triangular)
 */
__host__ __device__
int index(int i, int j, int n) {
	return n*i + j - ((i + 2)* (i + 1) / 2);
}

/* 
 * reads from input file a graph
 */
void read_input_file(char* inputfilename, bool* adjacencies_matrix) {
	FILE *inputfile;
	inputfile = fopen(inputfilename, "r");
	// tests if file exists 
	if (inputfile == NULL) {
		fprintf(stderr, "Can't open input file ConvexHullInput\n");
		exit(1);
	}
	int i, u, v, n, k, temp;
	fscanf(inputfile, "%d %d", &n, &k);
	std::cout << "n  " << n << "\n";
	for (i = 0; i < NUM_ADJ; i++)
		adjacencies_matrix[i] = false;

	// while loop that reads the data from the file
	while (!feof(inputfile)) {
		if (fscanf(inputfile, "%d %d", &u, &v) != 2)
			break;
		if (u > v) {
			temp = u; u = v; v = temp;
		}
		adjacencies_matrix[index(u, v, N)] = true;
		std::cout << "u and v " << u << "  " << v << "\n";
		std::cout << "adj " << adjacencies_matrix[index(u, v, N)] << "\n";
	}
	fclose(inputfile);
	//return adjacencies_matrix;
}

/*
 * functor generates random 0 and 1's depending on a certain probability
 */
struct Random_graph_gen {
	float prob;
	Random_graph_gen(float _prob) : prob(_prob) {}

	__host__ __device__
		bool operator () (int idx)
	{
		thrust::default_random_engine randEng;
		thrust::uniform_real_distribution<float> uniDist(0, 1);
		randEng.discard(idx); 
		return (uniDist(randEng) < prob);
	}
};

/*
 * generates a graph graph with a certain density defined by the parameter 
 * edges_percentage
 */
thrust::host_vector<bool> Random_graph(float edges_percentage){
	thrust::host_vector<bool> adj(NUM_ADJ);
	int i;
	thrust::transform(
		thrust::make_counting_iterator(0),
		thrust::make_counting_iterator(NUM_ADJ),
		adj.begin(),
		Random_graph_gen(edges_percentage));
	return adj;
}

/*
 * this function generates a random graph,
 * reduces the size of this graph, deleting vertices that can not
 * possibly be in the clique, and organizes the vertices by degree
 */
thrust::tuple<thrust::host_vector<bool>, thrust::host_vector<int>>
	reduced_organized_random_graph(float edges_percentage){

	thrust::host_vector<bool> adj = Random_graph(edges_percentage);
	thrust::host_vector<bool> active_vertex(N, true);
	thrust::host_vector<int> degree(N, -1);
	int i, j;
	
	int  n = 0, sum = 0, num_vertex = N, num_edges = 0, e = 0;
	int u, v, edge;
	while (n != num_vertex) {
		n = num_vertex;
		num_vertex = 0;
		num_edges = 0;
		for (i = 0; i < N; i++) {
			if (active_vertex[i]) {
				for (j = 0; j < N; j++) {
					if (active_vertex[j]) {
						if (i < j) {
							if (adj[index(i, j, N)]) {
								sum++;
								num_edges++;
							}
						}
						else if (j < i)
							if (adj[index(j, i, N)]) sum++;
					}
				}
				active_vertex[i] = (sum >= K);
				if (sum >= K) num_vertex++;
			}
			degree[i] = sum;
			sum = 0;
		}
	}
	thrust::host_vector<bool> result_adjacencies(NUM_ADJ, false);
	thrust::host_vector<int> result_list;

	u = 0; v = 0, num_edges = 0, e = 0;
	for (i = 0; i < N; i++){
		if (active_vertex[i]) {
			for (j = 0; j < N; j++){
				if (active_vertex[j]) {
					result_adjacencies[index(u, v, N)] = adj[index(i, j, N)];
					if (adj[index(i, j, N)])
						result_list.push_back(i * N + j);
					v++;
				}
			}
			v = 0;
			u++;
		}
	}
	return thrust::make_tuple(result_adjacencies, result_list);
}

/*
thrust::tuple<thrust::host_vector<bool>, thrust::host_vector<int>> 
	other_Random_graph(float edges_percentage){
	thrust::default_random_engine randEng;
	thrust::uniform_real_distribution<float> uniDist(0, 1);

	int i, j, n = 0, sum = 0, num_vertex = N, num_edges = 0;
	bool adj[N][N] = { false };
	bool active_vertex[N] = { true };
	int list[N*N] = { -1 };
	bool new_adj[N][N] = { false };

	for (i = 0; i < N; i++) {
		for (j = i + 1; j < N; j++) {
			randEng.discard(i*n + j);
			if (uniDist(randEng) < edges_percentage){
				sum++;
				adj[i][j] = true;
				adj[j][i] = true;
				num_edges++;
			}
		}
	}
	int u, v, edge;
	while (n != num_vertex) {
		n = num_vertex;	
		num_vertex = 0;

		for (i = 0; i < N; i++)
			active_vertex[i] = true;

		for (i = 0; i < n; i++) {
			for (j = 0; j < n; j++)
				if (adj[i][j]) sum++;
			active_vertex[i] = (sum >= K);
			if (sum >= K) num_vertex++;
			sum = 0;
		}

		u = 0; v = 0; edge = 0;

		for (i = 0; i < n; i++){
			if (active_vertex[i]) {
				for (j = 0; j < n; j++){
					if (active_vertex[j]) {
						new_adj[u][v] = adj[i][j];
						edge++;
						v++;
					}
				}
				v = 0;
				u++;
			}
		}
		
		for (i = 0; i < N; i++)
			for (j = 0; j < N; j++) {
				adj[i][j] = new_adj[i][j];
				new_adj[i][j] = false;
			}
	}
		
	edge = 0;
	thrust::host_vector<bool> result_adjacencies(num_vertex * (num_vertex - 1) / 2);
	thrust::host_vector<int> result_list(num_edges);
	for (i = 0; i < num_vertex; i++)
		for (j = i + 1; j < num_vertex; j++) {
			result_adjacencies[index(i, j, num_vertex)] = adj[i][j];
			if (adj[i][j]) {
				result_list[edge++] = i*N + j;
			}
		}
	return thrust::make_tuple(result_adjacencies, result_list);

}
*/

/*
 * calculates a partial factorial: n*(n-1)*...*(n-k+1)
 */
__host__ __device__
long long int partial_factorial(int n, int k) {
	int i;
	long long fac = 1;
	for (i = n - k + 1; i <= n; i++){
		fac = fac*i;
	}
	return fac;
}

/*
* calculates combination n chose k
*/
__host__ __device__
long long int combination(int n, int k){
	long long int p1 = partial_factorial(n, k);
	long long int p2 = partial_factorial(k, k);
	std::cout << "here";
	return p1 / p2;
}

__host__ __device__
long long int choose(int n, int k) {
	if (k == 0) return 1;
	return (n * choose(n - 1, k - 1)) / k;
}
/*
 * ferrament that decodes from a given number, which combination N chose K
 * we have
 */
__host__ __device__
thrust::host_vector<int> mthCombination(long long int mth, int size_subset){
	int k = size_subset, n = N;
	long long int index = mth, combination_n_k;
	int i = 0, list_index = 0, element = 0;
	thrust::host_vector<int> list(size_subset);
	while (k > 0) {
		if (k == 0) break;
		if (k == n) {
			for (i = list_index; i < K; i++)
				list[i] = element++;
			break;
		}
		combination_n_k = choose(n - 1, k - 1);
		if (index < combination_n_k) {
			k--;
			list[list_index++] = element;
		}
		else {
			index -= combination_n_k;
		}
		n--;
		element++;
	}
	return list;
	
}

/*
bool if_clique(int clique_id, bool* adjacent){
	std::cout << clique_id << " \n";
	int i;
	for (i = 0; i < NUM_ADJ; i++)
		std::cout << adjacent[i] << " adj\n";
	int vertex[K]; 
	mthCombination(clique_id, vertex);
	for (i = 0; i < K; i++)
		std::cout << vertex[i] << " ver\n";
	int u, v;
	for (v = 1; v < K; v++){
		for (u = 0; u < v; u++) {
			std::cout << vertex[u] << " " << vertex[v] << " " << index(vertex[u], vertex[v], N) << " " << adjacent[index(vertex[u], vertex[v], N)] << "  \n";
			if (!adjacent[index(vertex[u], vertex[v], N)]) {
				std::cout << vertex[u] << " " << vertex[v] << " " << " false \n";
				return false;
			}
		}
	}
	return true;
}
*/


bool recursive_find_clique(int vertex, bool* list_vertex, int k, thrust::host_vector<bool> adjacencies) {
	std::cout << " edge " << vertex << " k " <<  k << " \n";
	if (k == 0)
		return true;

	int i, u = vertex, v, vertex_degree = 0;
	bool new_list[N] = { false };
	for (v = u + 1; v < N; v++) {
		if (list_vertex[v] && adjacencies[index(u, v, N)]) {
			new_list[v] = true;
			vertex_degree++;
		}
	}
	if (vertex_degree < k)
		return false;

	bool is_in_a_clique = false;
	for (v = u + 1; v < N; v++) {
		if (new_list[v]) {
			if (recursive_find_clique(v, new_list, k - 1, adjacencies)) {
				is_in_a_clique = true;
				break;
			}
		}
	}
	return is_in_a_clique;
}

bool f_clique(int edge, thrust::host_vector<bool> adjacencies) {
	int u = 0, v = 0;
	v = edge % N;
	if (edge > v)
		u = (edge - v) / N;
	if (u > v) {
		int temp = v; v = u; u = temp;
	}
	//std::cout << " edge " << edge << " u " << u << " v " << v << " \n";
	bool possible_clique[N] = { false };
	int j, k = K - 2, num_vertices = N;

	//first layer: all vertexes greater than v that connect with u and v
	for (j = v + 1; j < num_vertices; j++)
		if (adjacencies[index(u, j, num_vertices)]) {
			possible_clique[j] = true;
		}

	return recursive_find_clique(v, possible_clique, k, adjacencies);
}



// functor that tests if is indeed a clique
struct is_a_clique {
	bool* adjacent;
	int* set_of_vertices;
	__host__ __device__
		is_a_clique(bool* _adjacent, int* _set_of_vertices) :
		adjacent(_adjacent), set_of_vertices(_set_of_vertices	) {}

	__host__ __device__
		int idx(int i, int j, int n) {
		return n*i + j - ((i + 2)* (i + 1) / 2);
	}

	__host__ __device__
		bool operator()(long long int clique_id){
		int list[K];
		int k = K, n = NUM_TREADS;
		long long int combination_n_k, index = clique_id;
		int i = 0, list_index = 0, element = 0;
		while (k > 0) {
			if (k == 0) break;
			if (k == n) {
				for (i = list_index; i < K; i++)
					list[i] = set_of_vertices[element++];
				break;
			}
			combination_n_k = choose(n - 1, k - 1);
			if (index < combination_n_k) {
				k--;
				list[list_index++] = set_of_vertices[element];
			}
			else {
				index -= combination_n_k;
			}
			n--;
			element++;
		}

		int u, v;
		for (v = 1; v < K; v++)
			for (u = 0; u < v; u++) 
				if (!adjacent[idx(list[u], list[v], N)])
					return false;
		return true;
	}
};

struct find_clique_from_edge{
	bool* adjacencies;
	__host__ __device__
		find_clique_from_edge(bool* _adjacencies) :
		adjacencies(_adjacencies) {}

	__host__ __device__
	bool recursive_find_clique(int vertex, bool* list_vertex, int k) {
		if (k == 0)
			return true;

		int i, u = vertex, v, vertex_degree = 0;
		bool new_list[N] = { false };
		for (v = u + 1; v < N; v++) {
			if (list_vertex[v] && adjacencies[index(u, v, N)]) {
				new_list[v] = true;
				vertex_degree++;
			}
		}
		if (vertex_degree < k)
			return false;

		bool is_in_a_clique = false;
		for (v = u + 1; v < N; v++) {
			if (new_list[v]) {
				if (recursive_find_clique(v, new_list, k - 1)) {
					is_in_a_clique = true;
					break;
				}
			}
		}
		return is_in_a_clique;
	}


	__host__ __device__
		bool operator() (int edge){
		int u = 0, v = 0;
		v = edge % N;
		if (edge > v)
			u = (edge - v) / N;
		if (u > v) {
			int temp = v; v = u; u = temp;
		}
		//std::cout << " edge " << edge << " u " << u << " v " << v << " \n";
		bool possible_clique[N] = { false };
		int j, k = K - 2, num_vertices = N;

		//first layer: all vertexes greater than v that connect with u and v
		for (j = v + 1; j < num_vertices; j++)
			if (adjacencies[index(u, j, num_vertices)]) {
				possible_clique[j] = true;
			}

		return recursive_find_clique(v, possible_clique, k);
	}
};

int main(void) {


	clock_t  clock1, clock2;
	clock1 = clock();
	int i, j;
	bool resres = false;
/*	long long int possible_cliques = choose(NUM_TREADS, K);
	long long int initial_combinations = choose(N, NUM_TREADS);
	//read_input_file("CliqueGraph.txt", adjacencies);
	thrust::device_vector<bool> adj = Random_graph(GRAPH_DENSITY);
	std::cout << " n " << N << " choose K " << NUM_TREADS << " " << initial_combinations << " \n";
	std::cout << " n " << NUM_TREADS << " choose K " << K << " " << possible_cliques << " \n";
	long long int it;

	for (it = 0; it < initial_combinations; it++) {

		thrust::device_vector<int> possible_clique(possible_cliques);
		thrust::device_vector<bool> result1(possible_cliques, false);
		thrust::sequence(possible_clique.begin(), possible_clique.end());

		thrust::device_vector<int> set_of_vertex = mthCombination(it, NUM_TREADS);
		
		thrust::transform(
			possible_clique.begin(),
			possible_clique.end(),
			result1.begin(),
			is_a_clique(thrust::raw_pointer_cast(&adj[0]),
			thrust::raw_pointer_cast(&set_of_vertex[0])));
		resres = thrust::any_of(
			result1.begin(), result1.end(), thrust::identity<bool>());
		if (resres) break;
	}
	std::cout << resres << " res \n";
	clock2 = clock();
	std::cout << "\nTime ALG1 " << (float)(clock2 - clock1) / CLOCKS_PER_SEC << " \n";
	//*/
	std::cout << "N " << N << "K " << K << "GRAPH_DENSITY " << GRAPH_DENSITY << " \n";
	clock1 = clock();
	int	possible_combinations_k;
	j = 0;
	
	thrust::tuple<thrust::host_vector<bool>, thrust::host_vector<int>> 
		lists = reduced_organized_random_graph(GRAPH_DENSITY);
	thrust::device_vector<bool> adjacencies = thrust::get<0>(lists);
	thrust::device_vector<int> list_adjacencies = thrust::get<1>(lists);
	thrust::device_vector<bool> result2(list_adjacencies.size());

	thrust::transform(
		list_adjacencies.begin(),
		list_adjacencies.end(),
		result2.begin(),
		find_clique_from_edge(thrust::raw_pointer_cast(&adjacencies[0]))
		);

	 bool resres2 = thrust::any_of(result2.begin(), result2.end(), thrust::identity<bool>());
	 std::cout << resres2 << " res \n";

	clock2 = clock();
	std::cout << "\nTime ALG2 " << (float)(clock2 - clock1) / CLOCKS_PER_SEC << " \n";
	//*/
	return 0;

}


