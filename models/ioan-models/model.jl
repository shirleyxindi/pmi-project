module Model

export AbstractModel, predict

"""
    AbstractModel

Abstract base type for all models.
"""
abstract type AbstractModel end

"""
    predict(model::AbstractModel, x)

Make a prediction using the model.

# Arguments
- `model::AbstractModel`: The model to use for prediction
- `x`: Input data

# Returns
The prediction result
"""
function predict(model::AbstractModel, x)
    error("predict not implemented for $(typeof(model))")
end

end # module Model
