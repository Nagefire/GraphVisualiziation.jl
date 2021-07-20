module GraphVisualization

import Makie: Scene, linesegments!, scatter!, meshscatter!,
                         arrows!, text!

import Makie.Colors: @colorant_str

import ArnoldiMethod: LR

import GeometryBasics: Circle, Sphere, Point, Point2, Point3

using GLMakie

import LightGraphs: AbstractGraph, adjacency_matrix, laplacian_matrix,
                    is_directed, edges, degree, src, dst, nv, ne

import LightGraphs.LinAlg: eigs

import LinearAlgebra: eigen

import NetworkLayout: buchheim, sfdp, shell, spectral, spring

import SparseArrays: SparseMatrixCSC, sparse

export gplot, gplot!, gplot3d, gplot3d!, spectral_layout, tree_layout,
       shell_layout, circular_layout

# Aliases for layouts
tree_layout=buchheim

sfdp_layout=sfdp

shell_layout=shell

spectral_layout=spectral

spring_layout=spring

"""
	gplot(G) → scene

Returns a Makie Scene containing a plot of the graph.

# Optional Keyword Arguments
- `layout` - The graph algorithm to use for determining vertex placement, defaults to `sfdp_layout`.
- `resolution` - The size/resolution of the scene to be returned, defaults to `(1200, 900)`.
- `backgroundc` - The color of the scene to be returned, defaults to `"colorant"#202020"`.
- `nodefillc` - The color for each node, defaults to `colorant"green"`.
- `nodesize` - The radius for each node, defaults to `0.5`.
- `nodelabel` - Text labels for each node in `G`, defaults to `nothing`.
- `edgelinewidth` - The thickness for edges, defaults to `3`.
- `edgelinec` - Color of edges, defaults to `colorant"lightgrey"`.
"""
function gplot(G::AbstractGraph, points::Vector{Point2{R}};
               resolution::Tuple{Int, Int}=(1200, 900),
               backgroundc=colorant"#202020",
               keyargs...) where R <: Real
	res = Scene(resolution=resolution, show_axis=false,
	            backgroundcolor=backgroundc)

	gplot!(res, G, points; keyargs...)

	res
end

gplot(G::AbstractGraph; layout::Function=sfdp_layout, keyargs...)=
    gplot(G, layout(G); keyargs...)

"""
	gplot!(scene, G, points[, keyargs])

Plots `G` onto a new axis in `scene` using `x` and `y` as coordinates for
each vertex.
"""
function gplot!(scene::Scene, G::AbstractGraph, points::Vector{Point2{R}};
                nodelabel=nothing,
                nodefillc=colorant"green",
                nodesize=0.5,
                edgelinewidth=3,
                edgelinec=colorant"lightgrey") where R <: Real
	n = nv(G)
	size(points, 1) != n &&
	throw(ArgumentError("Each vertex must have an (x, y) coordinate"))

	# Scale positions based on number of vertices
	points *= n

	# Plot the edges
	if is_directed(G)
		vecs = _vecs2d(G, points, nodesize)
		arrows!(scene, vecs..., linewidth=edgelinewidth,
		        color=edgelinec)
	else
		lines = _lines2d(G, points)
		linesegments!(scene, lines, linewidth=edgelinewidth,
		              color=edgelinec)
	end

	# Plot the vertices as points
	scatter!(scene, points, color=nodefillc,
	         markersize=nodesize*100)

	if nodelabel !== nothing && length(nodelabel) == n
		textsize = nodesize

		for i=1:n
			text!(scene, "$(nodelabel[i])",
			      textsize=textsize, position=points[i])
		end
	end
end

gplot!(scene::Scene, G::AbstractGraph; layout::Function=sfdp_layout,
       keyargs...)=gplot!(scene, G, layout(G); keyargs...)

# Returns the a set of lines representing undirected edges
function _lines2d(G::AbstractGraph, points::Vector{Point2{R}}) where R <: Real
	res = fill(Point(0., 0.) => Point(0., 0.), ne(G))
	i = 1

	@inbounds @simd for e in collect(edges(G))
		res[i] = points[e.src] => points[e.dst]
		i += 1
	end

	res
end

# Create a set of arrows representing directed edges
# TODO: Compute the center-edge offset so that the arrows begin and meet
# at the boundary of each vertex
function _vecs2d(G::AbstractGraph, points::Vector{Point2{R}}, r::R) where R <: Real
	m = ne(G)
	pos = fill(Point(0., 0.), m)
	cmp = fill(Point(0., 0.), m)
	i = 1

	@inbounds @simd for e in collect(edges(G))
		pos[i] = points[e.src]
		cmp[i] = points[e.dst] - points[e.src]
		i += 1
	end

	pos, cmp
end

"""
	gplot3d(G; ...)

Returns a scene containing a 3d plot of `G`.

# Optional Keyword Arguments
- `layout` - algorithm to use for placing the vertices.
- `nodefillc` - the color of the vertices.
- `nodesize` - the size of the vertex markers.
- `edgelinewidth` - the thickness of the edges
- `edgelinec` - the color of the edges.
"""
function gplot3d(G::AbstractGraph, points::Vector{Point3{R}};
                 resolution::Tuple{Int, Int}=(1200, 900),
                 backgroundc=colorant"#202020",
                 keyargs...) where R <: Real
	res = Scene(resolution=resolution, backgroundcolor=backgroundc)

	gplot3d!(res, G, points; keyargs...)

	res
end

gplot3d(G::AbstractGraph; layout::Function=spectral_layout, keyargs...)=
	gplot3d(G, layout(Matrix(laplacian_matrix(G))); keyargs...)

"""
	gplot3d!(scene, G; ...)

Plots `G` onto the scene as a 3-dimensional figure.

# Optional Keyword Arguments
- `layout` - algorithm to use for placing the vertices.
- `nodefillc` - the color to be used for the vertices.
- `nodesize` - the size of the markers used for nodes.
- `edgelinewidth` - the thickness of the edges.
- `edgelinec` - the color of the edges.
"""
function gplot3d!(scene::Scene, G::AbstractGraph, points::Vector{Point3{R}};
                  nodefillc=colorant"green",
                  nodesize=0.3,
                  edgelinewidth=2.5,
                  edgelinec=colorant"lightgrey") where R <: Real
	n = nv(G)
	size(points, 1) != n && throw(ArgumentError("Each vertex must have an (x,y,z) coordinate"))

	# Scale up based on the number of vertices
	points *= n

	linesegments!(scene, [points[e.src] => points[e.dst] for e in edges(G)],
	              linewidth=edgelinewidth, color=edgelinec)
	meshscatter!(scene, points, color=nodefillc, markersize=nodesize)
end

gplot3d!(scene::Scene, G::AbstractGraph; layout::Function=spectral_layout,
         keyargs...)=
	gplot3d!(scene, G, layout(Matrix(laplacian_matrix(G))); keyargs...)

end
