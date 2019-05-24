---
header-includes: |
    \graphicspath{ {/Users/aburkov/Dropbox/ML_Book_In_Progress/} }
---

# HDBSCAN*

As I mentioned in Chapter 9, **HDBSCAN**\index{HDBSCAN} is the recommended clustering algorithm: it is relatively fast and it's also very intuitive, because it only has one hyperparameter $n$, the minimum number of elements a cluster can have. In practice, deciding on $n$ is much simpler than on the number of clusters $k$ in k-Means. In this appendix, I describe HDBSCAN*, the most recent formulation of the HDBSCAN algorithm.

The **core distance**\index{distance!core} $d_{core}(\mathbf{x}_p)$ for an example $\mathbf{x}_p$ is the distance[^ed]\index{distance!Euclidean} to its $n$-th nearest neighbour (including $\mathbf{x}_p$ itself).

[^ed]: Here and below we use Euclidean distance.

The **mutual reachability distance**\index{distance!mutual reachability}
between two examples $\mathbf{x}_p$ and $\mathbf{x}_q$ with respect to $n$ is defined as $d_{mreach}(\mathbf{x}_p, \mathbf{x}_q) \stackrel{\text{def}}{=} \max\left\{d_{core}(\mathbf{x}_p), d_{core}(\mathbf{x}_q), d(\mathbf{x}_p, \mathbf{x}_q)\right\}$.

The **mutual reachability graph**\index{mutual reachability graph} $G_n$ with respect to $n$ is a complete **graph**[^cg]\index{graph}, in which the examples $\mathbf{x}_i$ for $i=1,\ldots,N$, are nodes, and the weight associated with each edge is the mutual reachability distance between the respective pair of examples.

[^cg]: A graph is said complete when any pair of nodes has an edge between them.

For a complete graph with weighted edges, a **minimum spanning tree**\index{minimum spanning tree} (MST) is defined as a subset of the edges of the graph that connects all the nodes together, without forming cycles and with the minimum possible sum of edge weights. That is, MST is a spanning tree whose sum of edge weights is as small as possible. 

The HDBSCAN* starts by constructing the MST of $G_n$. Consider a toy example[^toy] in Figure \ref{fig:HDBSCAN-toy-example}. In this example you see an MST computed for a two-dimensional dataset of $14$ examples.

\begin{figure}[H]
\centering
\includegraphics[width=0.85\textwidth]{Illustrations/HDBSCAN-toy-example}
\caption{Examples (filled circles) and edges of an MST computed over the space of mutual reachability distances with $n = 3$ and Euclidean distances (solid lines). Edge weights are omitted for the sake of clarity.}
\label{fig:HDBSCAN-toy-example}
\end{figure}

[^toy]: The illustrative example is taken from "Hierarchical Density Estimates for Data Clustering, Visualization, and Outlier Detection" by Campello et al. (2015). Used with permission.

Let $\epsilon$ denote the weight of an edge in the MST. The HDBSCAN* algorithm proceeds by removing edges of the MST one by one, starting with the edges having the highest weight. The rationale for proceeding this way is that the higher $\epsilon$ of some edge the farther two potential clusters joined by this edge located to one another according to the mutual reachability distance of the two closest examples belonging to two potential clusters. Removing an edge results in splitting $G_n$ into two connected components, and the dataset into two potential clusters the most distant from one another.

The table in Figure \ref{fig:HDBSCAN-table} shows the hierarchy of potential clusters obtained by removing edges of the MST in Figure \ref{fig:HDBSCAN-toy-example} one by one and the corresponding values of the weight $\epsilon$.

Look at the table in Figure \ref{fig:HDBSCAN-table}. It's useful for understanding of what HDBSCAN* will do next if you imagine that $\epsilon$ is gradually reduced from $\infty$ to $0$. Once the the weight of some edge of the MST becomes less than $\epsilon$, this edge is removed and one potential cluster is split into two new potential clusters. In the table in Figure \ref{fig:HDBSCAN-table}, there's only one cluster, identified as $1$, while $\epsilon \geq 7.1$. Once $\epsilon = 7.1$, we remove the edge between $\mathbf{x}_{10}$ and $\mathbf{x}_{11}$ and while $7.1 > \epsilon \geq 6.51$, there are two potential clusters identified as $2$ and $3$.

\begin{figure}[H]
\makebox[\textwidth]{\makebox[1.10\textwidth]{%
\centering
\includegraphics[width=1.10\textwidth]{Illustrations/HDBSCAN-table}}}
\caption{The hierarchy of potential clusters obtained by removing edges of the MST in Figure \ref{fig:HDBSCAN-toy-example}.}
\label{fig:HDBSCAN-table}
\end{figure}

\begin{figure}[H]
\centering
\includegraphics[width=0.60\textwidth]{Illustrations/HDBSCAN-stability}
\caption{Stability of the clusters in the hierarchy of clusters for the dataset in Figure \ref{fig:HDBSCAN-toy-example}.}
\label{fig:HDBSCAN-stability}
\end{figure}

By continuing gradually reduce $\epsilon$, we obtain an hierarchy or a tree of clusters. You can see that after removing some edges, such as the edge between $\mathbf{x}_{6}$ and $\mathbf{x}_{10}$ when $\epsilon$ decreases below $6.51$, some examples become isolated and are considered noise (idetified with zeroes).

Now the only remaining problem is to transform a hierarchy of clusters into a flat clustering, similar to the one obtained by k-Means. To obtain flat clustering, HDBSCAN* assigns a *stability* value to each potential cluster in the hierarchy. Intuitively, the stability of a cluster in the hierarchy reflects how long, while $\epsilon$ decreases, the cluster remains constant (unsplit). The stability $S(\mathcal{C}_k)$ of a cluster $\mathcal{C}_k$ is defined as follows,

$$ S(\mathcal{C}_k) \stackrel{\text{def}}{=} \sum_{\mathbf{x} \in \mathcal{C}_k}\left[ \frac{1}{\epsilon_{min}(\mathbf{x}, \mathcal{C}_k)} - \frac{1}{\epsilon_{max}(\mathcal{C}_k)}\right] $$

where $\epsilon_{max}(\mathcal{C}_k)$ is the value of $\epsilon$ at which $\mathcal{C}_k$ exists, $\epsilon_{min}(\mathbf{x}, \mathcal{C}_k)$ is the value of $\epsilon$ below which example $\mathbf{x}$ no longer belongs to cluster $\mathcal{C}_k$.

Figure \ref{fig:HDBSCAN-stability} shows the hiearachy of clusters for the dataset in Figure \ref{fig:HDBSCAN-toy-example} with their corresponding values of stability.

\begin{figure}[H]
\centering
\includegraphics[width=0.85\textwidth]{Illustrations/HDBSCAN-toy-example-clusters}
\caption{Flat clustering for the dataset in Figure \ref{fig:HDBSCAN-toy-example-clusters}.}
\label{fig:HDBSCAN-toy-example-clusters}
\end{figure}

In HDBSCAN*, the task of finding the "best" flat clustering is formulated as an optimization problem with the objective of maximizing the overall aggregated stabilities of the extracted clusters, in the following way:

$$ \max_{\delta_2, \ldots, \delta_K} J = \sum_{k=2}^K \delta_kS(\mathcal{C}_k)$$

$$ \text{subject to } \left \{
  \begin{aligned}
    &\delta_k\in\{0,1\},\; k=1,\ldots,K \\
    &\text{exactly one } \delta_{(\cdot)} = 1 \text{ in each path from a leaf cluster to the root}
  \end{aligned} \right. $$
  
For our example in Figure \ref{fig:HDBSCAN-stability}, $J$ is maximixed when the flat clusters are $\mathcal{C}_3, \mathcal{C}_4$, and $\mathcal{C}_5$ with $J = 7.69$. The final flat clustering is shown in Figure \ref{fig:HDBSCAN-toy-example-clusters}. Note that the elliptical form was chosen for illustrative purposes only: contrary to k-Means clusters, the clusters found by HDBSCAN* don't have any specific shape and only defined by the examples belonging to it. You can see in Figure \label{fig:HDBSCAN-toy-example-clusters} that examples $\mathbf{x}_4$ and $\mathbf{x}_{10}$ don't belong to any cluster and considered noise (or outliers).