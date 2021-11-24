#include<iostream>
#include<fstream>
#include<string>
#include<vector>
#include<map>
#include<unordered_set>
#include<unordered_map>
#include<numeric>
#include <algorithm>
#include<set>
#include<random>

#include <chrono> 
#include <filesystem>
#include <cassert>
#include "Source.h"


using namespace std;
namespace fs = std::filesystem;

std::random_device rd;
std::mt19937 gen(rd());


typedef double Weight_t;
const Weight_t MAX_WEIGHT_T = 100000000;
constexpr Weight_t DEFAULT_BEST_PATH = -1;
constexpr size_t DEFAULT_WEIGHT_FREE_ITEM = 1; //weight of symmetrical difference
constexpr size_t DEFAULT_WEIGHT_CAPACITY_DIFFERENCE = 1; //weight of capacity difference



template <typename T>
struct hash<std::pair<T, T>>
{
  std::size_t operator() (const std::pair<T, T>& pair) const {
    return std::hash<T>()(pair.first) ^ std::hash<T>()(pair.second);
  }
};


typedef pair<size_t, size_t> Edge_t;
typedef pair<Weight_t, Weight_t> Item_t; //weight/price

struct Knapsack_t {
  vector<vector<size_t>> conflict_matrix;
  vector<Item_t> items_list;
  Weight_t capacity;
  size_t insert_item(const Weight_t& iw, const Weight_t& ip) {
    conflict_matrix.push_back({});
    items_list.push_back(make_pair(iw, ip));
    return items_list.size() - 1;
  }
  void add_conflict(const size_t& item1, const size_t& item2) {
    conflict_matrix[item1].push_back(item2);
    conflict_matrix[item2].push_back(item1);
  }  
  size_t vertex_count() {
    return conflict_matrix.size();
  }
};


/// <summary>
///Structure for maintaining state of DP. 
/// </summary>
/// <typeparam name="T">Boolean for unweighted variant, int/float for weighted variant</typeparam>
struct State_t {
  unordered_set<size_t> free_items;
  Weight_t residual_capacity;
  State_t() : residual_capacity(0), free_items({}) {}
  State_t(const size_t& vertex_count, const size_t& res_capacity) : residual_capacity(res_capacity), free_items({}) {
    for (size_t i = 0; i < vertex_count; ++i) {
      free_items.insert(i);
    }
  }
  State_t(const unordered_set<size_t>& freeit, const size_t& res_cap) : free_items(freeit), residual_capacity(res_cap) {}
  State_t(const State_t& init_state) : free_items(init_state.free_items), residual_capacity(init_state.residual_capacity) {}
};

template <>
struct hash<State_t>
{
  std::size_t operator() (const State_t& state) const {
    size_t computed_hash = 31 * std::hash<Weight_t>()(state.residual_capacity);
    for (const size_t& free_item : state.free_items) {
      computed_hash += hash<size_t>()(free_item);
    }
    return computed_hash;
  }
};



/// <summary>
/// Decision node which hase two decisions decisionvariable = 0,1 and list of parrents (0 a 1)
/// </summary>
struct DecisionNode_t {
  Weight_t best_path;
  State_t node_state;
  vector<size_t> decisions;
  vector<vector<size_t>> parents;
  size_t decision_variable;
  DecisionNode_t() : node_state(), decisions(2, -1), parents({ {},{} }), best_path(DEFAULT_BEST_PATH), decision_variable(-1) { };
  DecisionNode_t(const size_t& vertex_count, const size_t& res_capacity, const size_t& dvar = DEFAULT_BEST_PATH, const Weight_t& bp = DEFAULT_BEST_PATH) :
    node_state(vertex_count, res_capacity), parents({ {},{} }), decision_variable(dvar), best_path(bp), decisions(2, -1) {}
  DecisionNode_t(const DecisionNode_t& dnode) : decision_variable(dnode.decision_variable),
    node_state(dnode.node_state), decisions(dnode.decisions), best_path(dnode.best_path), parents(dnode.parents){
    /*parents.resize(dnode.parents.size());
    for (size_t i = 0; i < dnode.parents.size(); ++i) {
      parents[i] = dnode.parents[i];
    }*/
  }
  DecisionNode_t(const State_t& init_state, const size_t& dvar= DEFAULT_BEST_PATH, const Weight_t& bp= DEFAULT_BEST_PATH) :
    node_state(init_state), parents({ {},{} }), decision_variable(dvar), best_path(bp), decisions(2, -1) {}
};


template <>
struct hash<DecisionNode_t>
{
  std::size_t operator() (const DecisionNode_t& node) const {
    return std::hash<State_t>()(node.node_state);
  }
};

inline bool operator==(const State_t& lhs, const State_t& rhs) { return lhs.residual_capacity == rhs.residual_capacity; }
inline bool operator!=(const State_t& lhs, const State_t& rhs) { return lhs.residual_capacity != rhs.residual_capacity; }
inline bool operator< (const State_t& lhs, const State_t& rhs) { return lhs.residual_capacity < rhs.residual_capacity; }
inline bool operator> (const State_t& lhs, const State_t& rhs) { return lhs.residual_capacity > rhs.residual_capacity; }
inline bool operator<=(const State_t& lhs, const State_t& rhs) { return lhs.residual_capacity <= rhs.residual_capacity; }
inline bool operator>=(const State_t& lhs, const State_t& rhs) { return lhs.residual_capacity >= rhs.residual_capacity; }


Weight_t state_distance(const State_t& lhs, const State_t& rhs, const size_t& weight_free_items_diff=DEFAULT_WEIGHT_FREE_ITEM, const size_t& weight_capacity_difference= DEFAULT_WEIGHT_CAPACITY_DIFFERENCE) {
  return weight_capacity_difference * abs(lhs.residual_capacity - rhs.residual_capacity) + weight_free_items_diff * (
    lhs.free_items.size() + rhs.free_items.size() - 2 * count_if(lhs.free_items.begin(), lhs.free_items.end(), [&rhs](size_t f_item) {
      return rhs.free_items.count(f_item); }));
}

typedef vector<DecisionNode_t> Layer_t;
typedef vector<Layer_t> DecisionDiagram_t;

/// <summary>
/// Counts the centroid of cluster. The free item appears in center if it is in at least half elements of cluster. The capacity is averaged.
/// </summary>
/// <param name="cluster"></param>
/// <param name="Decision_diagram_layer"></param>
/// <param name="centroid"></param>
void recompute_cluster_centroid(const vector<size_t>& cluster, const Layer_t& Decision_diagram_layer, State_t& centroid) {
  unordered_map<size_t, size_t> counting_map;
  centroid.residual_capacity = 0;
  for (const size_t& item : cluster) {
    for (const auto& item_free : Decision_diagram_layer[item].node_state.free_items) {
      if (!counting_map.count(item_free)) counting_map[item_free] = 1;
      else ++counting_map[item_free];
    }
    centroid.residual_capacity += Decision_diagram_layer[item].node_state.residual_capacity;
  }
  centroid.residual_capacity /= cluster.size();
  for (const auto& item : counting_map) {
    if (item.second > cluster.size() / 2) centroid.free_items.insert(item.first);
  }
}

/// <summary>
/// Recount the clusters for centroids
/// </summary>
/// <param name="clusters"></param>
/// <param name="Decision_diagram_layer"></param>
/// <param name="centroids"></param>
/// <param name="distances_cache"></param>
void assign_vecors_to_clusters(vector<vector<size_t>>& clusters, const Layer_t& Decision_diagram_layer, const vector<State_t>& centroids, unordered_map<pair<size_t, size_t>, Weight_t>& distances_cache) {
  clusters.clear();
  clusters.resize(centroids.size());
  for (size_t node_id = 0; node_id < Decision_diagram_layer.size(); ++node_id) {
    Weight_t min_weight = MAX_WEIGHT_T;
    size_t closest_centroid_id = 0;
    for (size_t centroid_id = 0; centroid_id < centroids.size(); ++centroid_id) {
      if (!distances_cache.count(make_pair(node_id, centroid_id))) {
        //cerr << "this should not happen and we have to recount the distance" << endl;
        distances_cache[make_pair(node_id, centroid_id)] = state_distance(Decision_diagram_layer[node_id].node_state, centroids[centroid_id]);
      }
      if (min_weight > distances_cache[make_pair(node_id, centroid_id)]) {
        min_weight = distances_cache[make_pair(node_id, centroid_id)];
        closest_centroid_id = centroid_id;
      }
    }
    clusters[closest_centroid_id].push_back(node_id);
  }
}

bool clusterize_vectors(unordered_map<pair<size_t, size_t>, Weight_t>& distances_cache, vector<vector<size_t>>& clusters, const Layer_t& Decision_diagram_layer, vector<State_t>& centroids){
  //first recompute the centroids
  bool change = false;
  for (size_t cluster_id = 0; cluster_id < centroids.size(); ++cluster_id) { //recompute clusters and update the distance matrix
    State_t new_centroid;
    recompute_cluster_centroid(clusters[cluster_id], Decision_diagram_layer, new_centroid);
    if (centroids[cluster_id] != new_centroid) {
      change = true;
      centroids[cluster_id] = new_centroid;
      
      for (size_t node_id = 0; node_id < Decision_diagram_layer.size(); ++node_id) {
        distances_cache[make_pair(node_id, cluster_id)] = state_distance(Decision_diagram_layer[node_id].node_state, new_centroid);
      }
    }
  }
  if (!change) return false;

  assign_vecors_to_clusters(clusters, Decision_diagram_layer, centroids, distances_cache);
  return true;
  //drop the clusters and make the clusters great again
}

size_t choose_next_init_centroid(vector<Weight_t>& distances_cache) {
  std::discrete_distribution<> d(distances_cache.begin(), distances_cache.end()); 
  return d(gen);
}

void init_centroids(unordered_map<pair<size_t, size_t>, Weight_t>& distances_cache, const Layer_t& Decision_diagram_layer, vector<State_t>& centroids, const size_t& k) {
  vector<Weight_t> closest_centroid(Decision_diagram_layer.size(), MAX_WEIGHT_T);
  while (centroids.size() < k) {
    size_t centroid_id = choose_next_init_centroid(closest_centroid);
    centroids.push_back(Decision_diagram_layer[centroid_id].node_state);
    for (size_t node_id = 0; node_id < Decision_diagram_layer.size(); ++node_id) {
      distances_cache[make_pair(node_id, centroids.size()-1)] = state_distance(Decision_diagram_layer[node_id].node_state, centroids.back());
      closest_centroid[node_id] = min(distances_cache[make_pair(node_id, centroids.size() - 1)] * distances_cache[make_pair(node_id, centroids.size() - 1)],closest_centroid[node_id]);
    }
  }
}

void kmeans_alg(Layer_t& Decision_diagram_layer, const size_t& k, vector<vector<size_t>>& clusters) {
  unordered_map<pair<size_t, size_t>,Weight_t> distances_cache; //distance of any vertex to any cluster centroid. second is centroid id in the centroids vector, first is the id of the item/node
  vector<State_t> centroids;
  if (k == Decision_diagram_layer.size()) {
    for (size_t i = 0; i < k; ++i) { clusters.push_back({ i }); }
    return;
  }
  init_centroids(distances_cache, Decision_diagram_layer, centroids, k);
  assign_vecors_to_clusters(clusters, Decision_diagram_layer, centroids, distances_cache);
  int i = 10;
  while (clusterize_vectors(distances_cache, clusters, Decision_diagram_layer, centroids) && --i > 0);
}

/// <summary>
/// update the neigbours of dominating vertex in the state
/// </summary>
/// <param name="input_instance"></param>
/// <param name="state_to_update"></param>
/// <param name="dominating_vertex"></param>
/// <returns>0 if the capacity is bellow zero, 1 otherwise</returns>
bool state_decision_update(Knapsack_t& input_instance, State_t& state_to_update, const size_t& inserted_item) {
  for (size_t& vertex_id : input_instance.conflict_matrix[inserted_item]) {
    state_to_update.free_items.erase(vertex_id);
  }
  if (state_to_update.residual_capacity < input_instance.items_list[inserted_item].first) return 0;
  state_to_update.residual_capacity -= input_instance.items_list[inserted_item].first;
  return 1;
}

/// <summary>
/// insert a new node to a new layer
/// </summary>
/// <param name="input_instance">input instance</param>
/// <param name="new_layer_hash">has table of new layer to save the cache</param>
/// <param name="DD_new_layer">new layer in representation, to insert new node and update parents</param>
/// <param name="source_decision_node">decision node for which we are making decision</param>
/// <param name="decision">decision value</param>
/// <param name="parent_index">index of the parent to add to parents[decision]</param>
/// <param name="descendat_variable_id">next decision variable id according to order of OBDD</param>
/// <returns></returns>
size_t DD_insert_new_node(Knapsack_t& input_instance, unordered_map<State_t, size_t>& new_layer_hash, vector<DecisionNode_t>& DD_new_layer, const DecisionNode_t& source_decision_node,
  const size_t& decision, const size_t& parent_index, const size_t& descendat_variable_id) {
  size_t decision_variable_id = source_decision_node.decision_variable;
  Weight_t new_best_path = source_decision_node.best_path;
  if (decision) new_best_path += input_instance.items_list[decision_variable_id].second;
  State_t new_state(source_decision_node.node_state);

  
  if (decision && 
    (!source_decision_node.node_state.free_items.count(decision_variable_id) ||
          !state_decision_update(input_instance, new_state, decision_variable_id))) return -1; //update state if decision x_i=1, item we want to insert is free and the capacity of item is less than residual capacity


  // find even subset => subset and restriction are dominated
  auto new_layer_inserter = new_layer_hash.insert(make_pair(new_state, DD_new_layer.size())); //if inserted, it will point to new layer end vertex, which we will add 
  if (new_layer_inserter.second == true) { //was not inserted did not exist, need to bye inserted
    DD_new_layer.push_back(DecisionNode_t(new_state, descendat_variable_id, new_best_path)); //insert new node, dont fill parents, just variable number
  }
  DD_new_layer[new_layer_inserter.first->second].parents[decision].push_back(parent_index); // add parent for the node together with its decision edge
  DD_new_layer[new_layer_inserter.first->second].best_path = min(new_best_path, DD_new_layer[new_layer_inserter.first->second].best_path);
  return new_layer_inserter.first->second;
}

/// <summary>
/// Swap two nodes and all connections on layer
/// </summary>
/// <param name="Decision_diagram"></param>
/// <param name="layer">Layer of swaping nodes</param>
/// <param name="node1">First node to swap on specified layer</param>
/// <param name="node2">Second node to swap on specified layer</param>
void DD_swap_nodes(DecisionDiagram_t& Decision_diagram, const size_t& layer, const size_t& node1, const size_t& node2) {
  if (node1 == node2) return;
  //flip pointers from parents and descendants node2 <-> node1 for each decision 0,1
  for (size_t decision_val = 0; decision_val < 2; ++decision_val) {
    for (size_t& parent : Decision_diagram[layer][node1].parents[decision_val]) { //for each parent of node1 with parent[decision_val]=node1 redirect it to node2
      assert(Decision_diagram[layer - 1][parent].decisions[decision_val] == node1);
      Decision_diagram[layer - 1][parent].decisions[decision_val] = node2;
    }
    for (size_t& parent : Decision_diagram[layer][node2].parents[decision_val]) { //for each parent of node 2 do the same as above
      assert(Decision_diagram[layer - 1][parent].decisions[decision_val] == node2);
      Decision_diagram[layer - 1][parent].decisions[decision_val] = node1;
    }
    if (Decision_diagram[layer][node1].decisions.size() > decision_val && Decision_diagram[layer][node1].decisions[decision_val] != -1) { //we have already instanciated the decision an it is not null
      replace(Decision_diagram[layer + 1][Decision_diagram[layer][node1].decisions[decision_val]].parents[decision_val].begin(), 
        Decision_diagram[layer + 1][Decision_diagram[layer][node1].decisions[decision_val]].parents[decision_val].end(),
        node1, node2);
    }
    if (Decision_diagram[layer][node2].decisions.size() > decision_val && Decision_diagram[layer][node2].decisions[decision_val] != -1) { //we have already instanciated the decision an it is not null
      replace(Decision_diagram[layer + 1][Decision_diagram[layer][node2].decisions[decision_val]].parents[decision_val].begin(), 
        Decision_diagram[layer + 1][Decision_diagram[layer][node2].decisions[decision_val]].parents[decision_val].end(),
        node2, node1);
    }
  }
  //flip content of nodes
  //swap states
  Decision_diagram[layer][node1].node_state.free_items.swap(Decision_diagram[layer][node2].node_state.free_items);
  swap(Decision_diagram[layer][node1].node_state.residual_capacity, Decision_diagram[layer][node2].node_state.residual_capacity);
  //swap parents
  Decision_diagram[layer][node1].parents[0].swap(Decision_diagram[layer][node2].parents[0]);
  Decision_diagram[layer][node1].parents[1].swap(Decision_diagram[layer][node2].parents[1]);
  //flip descendants
  //Decisions_swap
  Decision_diagram[layer][node1].decisions.swap(Decision_diagram[layer][node2].decisions);
  //best_path swap and decision_variable
  swap(Decision_diagram[layer][node1].best_path, Decision_diagram[layer][node2].best_path);
  swap(Decision_diagram[layer][node1].decision_variable, Decision_diagram[layer][node2].decision_variable);
}


/// <summary>
/// Removes a node on specified layer by swaping it with the last node of layer and removing it
/// </summary>
/// <param name="Decision_diagram">dd node for swap</param>
/// <param name="layer_id">layer id where, node_id should be deleted</param>
/// <param name="node_id">id of node on layer_id</param>
void DD_remove_one_node(DecisionDiagram_t& Decision_diagram, const size_t& layer_id, const size_t& node_id) {
  //first clear all refernces to node which we want to delete
  for (size_t decision_val = 0; decision_val < 2; ++decision_val) {
    //clear parent decisions
    for (auto parent : Decision_diagram[layer_id][node_id].parents[decision_val]) {
      Decision_diagram[layer_id - 1][parent].decisions[decision_val] = -1;
    }
    Decision_diagram[layer_id][node_id].parents[decision_val].clear();
    //clear descendant parents pointers
    if (Decision_diagram[layer_id][node_id].decisions.size() > decision_val  //we have created the decision for descendatn
      && Decision_diagram[layer_id][node_id].decisions[decision_val] != -1) { //the decision is not null
      auto element = find(Decision_diagram[layer_id + 1][Decision_diagram[layer_id][node_id].decisions[decision_val]].parents[decision_val].begin(),
        Decision_diagram[layer_id + 1][Decision_diagram[layer_id][node_id].decisions[decision_val]].parents[decision_val].end(), node_id);
      *element = Decision_diagram[layer_id + 1][Decision_diagram[layer_id][node_id].decisions[decision_val]].parents[decision_val].back();
      Decision_diagram[layer_id + 1][Decision_diagram[layer_id][node_id].decisions[decision_val]].parents[decision_val].pop_back();
      Decision_diagram[layer_id][node_id].decisions[decision_val] = -1;
    }

  }
  DD_swap_nodes(Decision_diagram, layer_id, node_id, Decision_diagram[layer_id].size() - 1); //swap node to delete and last vertex
  Decision_diagram[layer_id].pop_back(); //remove lasta vertexa
}



/// <summary>
/// Merge all nodes from vector nodes_to_merge on layer_id (last layer). We expect that nodes are on the last layer as we do nothing with the decisions. The new node has no outgoing decisions as it has to be determined in new refernece
/// </summary>
/// <param name="Decision_diagram">Input diagram.</param>
/// <param name="layer_id">Layer containing vertices to merge.</param>
/// <param name="nodes_to_merge">Vector of vertices to merge.</param>
void DD_merge_nodes(DecisionDiagram_t& Decision_diagram, const size_t& layer_id, vector<size_t>& nodes_to_merge) {
  //merge state => merge free items and save the bigger capacity
  Weight_t merged_capacity = Decision_diagram[layer_id][nodes_to_merge[0]].node_state.residual_capacity;
  Weight_t merged_best_path = Decision_diagram[layer_id][nodes_to_merge[0]].best_path;
  for (auto node_id : nodes_to_merge) {
    if (node_id == nodes_to_merge[0]) continue; //next line merges the state
    Decision_diagram[layer_id][nodes_to_merge[0]].node_state.free_items.insert(Decision_diagram[layer_id][node_id].node_state.free_items.begin(), Decision_diagram[layer_id][node_id].node_state.free_items.end());
    merged_capacity = max(merged_capacity, Decision_diagram[layer_id][node_id].node_state.residual_capacity);
    merged_best_path = max(merged_best_path, Decision_diagram[layer_id][node_id].best_path);
  }
  //computed result capacity
  Decision_diagram[layer_id][nodes_to_merge[0]].node_state.residual_capacity = merged_capacity;
  Decision_diagram[layer_id][nodes_to_merge[0]].best_path = merged_best_path;
  
  //general for DDs redirect parents pointers
  for (size_t decision_val = 0; decision_val < 2; ++decision_val) { //connect parrents and descendants to new vertex
    unordered_set<size_t> collect_parents; //we can have multiple parents
    for (auto node_id : nodes_to_merge) {
      collect_parents.insert(Decision_diagram[layer_id][node_id].parents[decision_val].begin(), Decision_diagram[layer_id][node_id].parents[decision_val].end());
      Decision_diagram[layer_id][node_id].parents[decision_val].clear();
      //check decisions of each node to merge, remove pointers betwenn node and descendatns
      if (Decision_diagram[layer_id][node_id].decisions.size() > decision_val && Decision_diagram[layer_id][node_id].decisions[decision_val] != -1) {
        Decision_diagram[layer_id + 1][Decision_diagram[layer_id][node_id].decisions[decision_val]].parents[decision_val].erase(remove(
          Decision_diagram[layer_id + 1][Decision_diagram[layer_id][node_id].decisions[decision_val]].parents[decision_val].begin(),
          Decision_diagram[layer_id + 1][Decision_diagram[layer_id][node_id].decisions[decision_val]].parents[decision_val].end(), node_id), 
          Decision_diagram[layer_id + 1][Decision_diagram[layer_id][node_id].decisions[decision_val]].parents[decision_val].end()); //remove pointer from descendant
        //TODO: remove deadend nodes
        Decision_diagram[layer_id][node_id].decisions[decision_val] = -1; //remove pointer to descendant
      }
    }
    Decision_diagram[layer_id][nodes_to_merge[0]].parents[decision_val].assign(collect_parents.begin(), collect_parents.end()); //assign parents to the first node, which we chosed to be merged
    if (Decision_diagram[layer_id][nodes_to_merge[0]].decisions.size() > decision_val) { //clear the decisions and have to create new one
      Decision_diagram[layer_id][nodes_to_merge[0]].decisions[decision_val] = -1;
    } 
    for (auto parent : collect_parents) { //set decisions for parents to right node
      Decision_diagram[layer_id - 1][parent].decisions[decision_val] = nodes_to_merge[0];
    }
    
  }
  //finally we remove all vertices which we do not want, we have to do it from the back as we change the order of nodes during the process
  /*for (int rwalker = nodes_to_merge.size() - 1; rwalker > 0; --rwalker) {
    DD_remove_one_node(Decision_diagram, layer_id, nodes_to_merge[rwalker]);
  }*/
}

//operator for my maxheap, comparing the second element, which identifies the node to delete
inline bool heap_compare(const pair<size_t, size_t>& lhs, const pair<size_t, size_t>& rhs) {
  return lhs.first < rhs.first;
}

size_t clear_clusters(DecisionDiagram_t& Decision_diagram, const size_t& layer_id, vector<vector<size_t>>& clusters_to_clear) {
  size_t orig_size = Decision_diagram[layer_id].size();
  vector<pair<size_t, size_t>> rear_max_heap; //(node_id, cluster_id)
  rear_max_heap.reserve(clusters_to_clear.size());
  for (size_t i = 0; i < clusters_to_clear.size(); ++i) {
    if (clusters_to_clear[i].size() > 1)
      rear_max_heap.push_back(make_pair(clusters_to_clear[i].back(), i));
  }
  make_heap(rear_max_heap.begin(), rear_max_heap.end(), heap_compare);
  while (rear_max_heap.size()) {
    pop_heap(rear_max_heap.begin(), rear_max_heap.end(), heap_compare);
    DD_remove_one_node(Decision_diagram, layer_id, rear_max_heap.back().first);
    clusters_to_clear[rear_max_heap.back().second].pop_back();
    if (clusters_to_clear[rear_max_heap.back().second].size() > 1) {
      rear_max_heap.back() = make_pair(clusters_to_clear[rear_max_heap.back().second].back(), rear_max_heap.back().second);
      push_heap(rear_max_heap.begin(), rear_max_heap.end(), heap_compare);
    } else {
      rear_max_heap.pop_back();
    }
  }
  return orig_size - Decision_diagram[layer_id].size();
}

/// <summary>
/// creates an exact decision diagram for a total weighted set problem
/// </summary>
/// <returns>best dominating set value</returns>
Weight_t create_dd(Knapsack_t& input_instance, vector<size_t>& variables_order, const size_t& k) {
  Weight_t best_yet = 0;
  //vector<bool> DDnode_is_dominated({ 0 }); //incidence whether we should ignore a node of exact dd
  if (variables_order.back() != -1)
    variables_order.push_back(-1); // terminal node is in the last layer, we dont care about the value
  DecisionDiagram_t exact_DD;
  exact_DD.push_back({ DecisionNode_t(input_instance.conflict_matrix.size(), input_instance.capacity , variables_order[0], 0) }); //root with clean state 
  for (size_t variable_order_index = 0; variable_order_index < variables_order.size()-1; ++variable_order_index) {//layer by layer
    //cerr << "<<<<<---------Layer " << variable_order_index << "-------";
    
    unordered_map<State_t, size_t> new_layer; //decision node and its index in the new layer
    //if (variable_order_index == exact_DD.size()) {
      exact_DD.push_back({});
    //} //init new layer
    for (size_t processed_node_index = 0; processed_node_index < exact_DD[variable_order_index].size(); ++processed_node_index) { //process each node on current layer
      //TODO:function which checks the extendibility. In case the node is too weak, we will kill it
      for (size_t new_decision = 0; new_decision < 2; ++new_decision) { //each decision for current node 
        exact_DD[variable_order_index][processed_node_index].decisions[new_decision] =                          //connect decision to the right node returned from this complicated function
          DD_insert_new_node(input_instance, new_layer, exact_DD[variable_order_index + 1], exact_DD[variable_order_index][processed_node_index],
            new_decision, processed_node_index, variables_order[variable_order_index + 1]);
      }
    }
    if (k < exact_DD.back().size()) {
      vector<vector<size_t>> clusters;
      kmeans_alg(exact_DD.back(), k, clusters);
      for (auto& cluster : clusters) {
        if (cluster.size() > 1) {
          DD_merge_nodes(exact_DD, variable_order_index+1, cluster); //remove afterwards
        }
      }
      clear_clusters(exact_DD, variable_order_index + 1, clusters);
    }
    //find_dead_end_nodes(exact_DD, DDnode_is_dominated, DDnodes_dominated_list); //we will find nodes which we dont want to expand and hence can be deleted
    //clear_last_constructed_layer(exact_DD, DDnodes_dominated_list); // clear last layer
    //just print the best path
    Weight_t best_solution = 0;
    for (auto fnode : exact_DD.back()) {
      best_solution = max(best_solution, fnode.best_path);
    }
    cerr << "<<<<<<<-------" << variable_order_index << ". layer " << exact_DD.back().size() << " nodes,  best value: " << best_solution << "----->>>>>>>>\n";
    //if (exact_DD.size() > 10) (exact_DD.end() - 2)->clear();
  }
  
  Weight_t best_solution = 0;
  for (auto fnode : exact_DD.back()) {
    best_solution = max(best_solution, fnode.best_path);
  }
  cerr << "<<<<< Best path value: " << best_solution << "-----State: ";
  return best_solution;
  
  //return 0;
};


void sort_variables_order(Knapsack_t& input_instance, vector<size_t>& variables_order) {
  sort(variables_order.begin(), variables_order.end(), [&input_instance](const size_t& item1_id, const size_t& item2_id) {
    return input_instance.items_list[item1_id].second * input_instance.items_list[item2_id].first > input_instance.items_list[item2_id].second * input_instance.items_list[item1_id].first;
    });
}


bool read_input_from_file(ifstream& input_stream, Knapsack_t& knapsack) {
  if (!input_stream.good()) {
    cerr << "Data loading error.\n";
    return 0;
  }
  size_t items_count, conflict_count, id, item1, item2;
  Weight_t k_capacity, weight, price;
  input_stream >> items_count >> conflict_count >> k_capacity;
  knapsack.capacity = k_capacity;
  //load items with its weight and price
  for (size_t i = 0; i < items_count; ++i) {
    input_stream >> id >> weight >> price;
    knapsack.insert_item(weight, price);
    
  }
  //load conflict graph
  for (size_t i = 0; i < conflict_count; ++i) {
    input_stream >> id >> item1 >> item2;
    knapsack.add_conflict(item1, item2);
  }
  return 1;
}

bool generate_rnd_instance(Knapsack_t& knapsack, const Weight_t& capacity, const size_t& items_count, 
                                                 const Weight_t& weight_LB, const Weight_t& weight_UB, 
                                                 const Weight_t& price_LB, const Weight_t& price_UB, const double& graph_density)
{
  knapsack.conflict_matrix.clear();
  knapsack.items_list.clear();
  knapsack.capacity = capacity;
  uniform_real_distribution<Weight_t> dis_weight(weight_LB, weight_UB), dis_price(price_LB, price_UB);
  knapsack.conflict_matrix.resize(items_count);
  while (knapsack.items_list.size() < items_count) {
    knapsack.items_list.push_back(make_pair(dis_weight(gen),dis_price(gen)));
  }
  size_t edge_count = max(0, (int)round(graph_density * items_count * (items_count - 1)));
  unordered_set<pair<size_t, size_t>> edges;
  size_t v_1, v_2;
  uniform_int_distribution <size_t> dis_edge(0, items_count-1);
  while (edges.size() < edge_count) {
    v_1 = dis_edge(gen);
    v_2 = dis_edge(gen);
    if (v_1 != v_2) edges.insert(make_pair(v_1, v_2));
  }
  
  for (const auto edge : edges) {
    knapsack.conflict_matrix[edge.first].push_back(edge.second);
    knapsack.conflict_matrix[edge.second].push_back(edge.first);
  }
  return 1;
}
int main() {
  string path = "input";
  ofstream results_stream(".\\output\\results.csv");
  results_stream << "n, max_width, result, time\n";
  for (const auto& entry : fs::directory_iterator(path)) {
    cout << entry.path() << endl;
    results_stream << entry.path() << endl;

    //load data
    Knapsack_t input_instance;
/*    ifstream input_stream(entry.path());
    if (!read_input_from_file(input_stream, input_instance)) {
      cerr << "Something went wrong." << endl;
    }
    input_stream.close();
    */
   
    generate_rnd_instance(input_instance, 600, 1000, 20, 100, 1, 100, 0.5);

    vector<size_t> variables_order;
    for (size_t i = 0; i < input_instance.items_list.size(); ++i) {
      variables_order.push_back(i);
    }

    sort_variables_order(input_instance, variables_order);
    //for (int i = 1; i <= 1; i *= 10) {
      //max width restrictions
    
      auto start = chrono::high_resolution_clock::now();
      Weight_t result = create_dd(input_instance, variables_order, 1000);
      auto duration = chrono::duration_cast<chrono::seconds>(chrono::high_resolution_clock::now() - start);
      cout << result << " in " << duration.count() << "s" << endl;
      results_stream << duration.count() << endl;
    //}
  }


  return 0;
}