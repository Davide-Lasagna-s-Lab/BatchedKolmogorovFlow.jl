# The tasks of the propagators on the CPU, each with a chunk of the fields of a batch.

# On a GPU one call advances a whole batch, and the device works on all its fields at once. On a
# CPU an operation on an array runs on one core, so that a batch, however large, uses one core.
# To use the others, a propagator made with `tasks` greater than one cuts a batch into as many
# chunks, ranges of whole fields of sizes as equal as can be, and gives one to each task. A batch
# of 10 fields with 4 tasks is cut as
#
#     task 1   fields 1:2
#     task 2   fields 3:5
#     task 3   fields 6:7
#     task 4   fields 8:10
#
# Every task advances its chunk as a batch of its own, on its own thread, at the same time as the
# others. The fields of a batch do not act on one another, a field with its perturbations being
# one segment of an orbit, so that the chunks are separate integrations and the result is that of
# one task. The tasks run at the same time and cannot share arrays or plans: each has its own
# workspaces, made with the propagator.

# the ranges of the fields of the chunks of B fields, as many as the tasks or as the fields if fewer
function _ranges(B::Int, tasks::Int)
    t = min(tasks, B)

    return [(B*(c - 1) ÷ t + 1):(B*c ÷ t) for c in 1:t]
end

# f(chunk) for every chunk, a named tuple of the range of its fields, `fields`, and of its
# workspaces, each on its task. Chunk c always runs on thread c, so that its workspaces are used by
# the thread that wrote them first and stay in the memory next to that core.
function _foreach_chunk(f, chunks::Vector)
    # ---- one chunk: the whole batch, with no threads ----
    length(chunks) == 1 &&
        return f(chunks[1])

    # ---- every chunk on its own thread ----
    Threads.@threads :static for chunk in chunks
        f(chunk)
    end

    return nothing
end
