export tree_layout, shell_layout, circular_layout, spring_layout,
       spectral_layout


function spectral_layout(G::AbstractGraph; dims::Int=2, weight::SparseMatrixCSC=nothing)
	if nv(G) == 1
		return [0.0], [0.0]
	elseif nv(G) == 2
		return [0.0, 1.0], [0.0, 0.0]
	end

	_spectral = if dims == 2 _spectral2d else _spectral3d end

	if weight == nothing
		weight = ones(ne(G))
	end

	if nv(G) > 500
		A = sparse(Int[src(e) for e in edges(G)],
		           Int[dst(e) for e in edges(G)],
			   weight, nv(G), nv(G))

		if is_directed(G)
			A += transpose(A)
		end

		return _spectral(A)
	else
		L = laplacian_matrix(G)
		return _spectral(Matrix(L))
	end
end

function _spectral2d(L::Matrix)
	eigvals, eigvecs = eigen(L)
	index = sortperm(eigvals)[2:3]

	return [Point2(real(eigvecs[i, index[1]]),
	               real(eigvecs[i, index[1]])) for i=1:size(A, 1)]
end

function _spectral2d(A::SparseMatrixCSC)
	data = vec(sum(A, dims=1))
	D = sparse(Base.OneTo(length(data)), Base.OneTo(length(data)), data)
	L = D - A
	eigenvalues, eigenvectors = eigs(L, nev=3, which=LR())
	index = sortperm(real(eigenvalues))[2:3]

	return [Point2(real(eigenvectors[i, index[1]]),
	               real(eigenvectors[i, index[1]])) for i=1:size(A, 1)]
end

# Note: A graph must have at least 4 vertices to use 3d spectral positioning
function _spectral3d(L::Matrix)
	eigvals, eigvecs = eigen(L)
	ind = sortperm(real(eigvals))[2:4]

	return [Point3(real(eigvecs[i, ind[1]]),
		       real(eigvecs[i, ind[2]]),
		       real(eigvecs[i, ind[3]])) for i=1:size(L, 1)]
end

function _spectral3d(A::SparseMatrixCSC)
	data = vec(sum(A, dims=1))
	D = sparse(Base.OneTo(length(data)), Base.OneTo(length(data)), data)
	L = D - A
	eigenvals, eigenvecs = eigs(L, nev=4, which=LR())
	ind = sortperm(real(eigenvals))[2:4]

	return [Point3(real(eigenvecs[i, ind[1]]),
	               real(eigenvecs[i, ind[2]]),
	               real(eigenvecs[i, ind[3]])) for i=1:size(A, 1)]
end
