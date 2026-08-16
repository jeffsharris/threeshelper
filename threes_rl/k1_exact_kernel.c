#include <stddef.h>
#include <stdint.h>

#define K1_BOARD_CELLS 16
#define K1_BOARD_SIDE 4
#define K1_ACTIONS 4
#define K1_MODELS 4
#define K1_PHASES 4
#define K1_PATTERNS 21
#define K1_SYMMETRIES 8
#define K1_MAX_PATTERN 6
#define K1_NUM_RANKS 16
#define K1_STARTER_TILE 1536
#define K1_TERMINAL_TILE 12288

int k1_kernel_abi_version(void) {
    return 1;
}

static int k1_can_merge(int32_t a, int32_t b) {
    if (a <= 0 || b <= 0) {
        return 0;
    }
    if ((a == 1 && b == 2) || (a == 2 && b == 1)) {
        return 1;
    }
    return a >= 3 && a == b && a < K1_TERMINAL_TILE;
}

static int32_t k1_merge_value(int32_t a, int32_t b) {
    if ((a == 1 && b == 2) || (a == 2 && b == 1)) {
        return 3;
    }
    return a * 2;
}

static int k1_line_index(int action, int lane, int position) {
    if (action == 0) {
        return position * K1_BOARD_SIDE + lane;
    }
    if (action == 1) {
        return (K1_BOARD_SIDE - 1 - position) * K1_BOARD_SIDE + lane;
    }
    if (action == 2) {
        return lane * K1_BOARD_SIDE + position;
    }
    return lane * K1_BOARD_SIDE + (K1_BOARD_SIDE - 1 - position);
}

int k1_base_move(
    const int32_t *board,
    int action,
    int32_t *out_board,
    int32_t *out_eligible_pairs,
    int32_t *out_eligible_count
) {
    int lane;
    int position;
    int eligible_count = 0;
    if (board == NULL || out_board == NULL || out_eligible_pairs == NULL ||
        out_eligible_count == NULL || action < 0 || action >= K1_ACTIONS) {
        return 1;
    }
    for (position = 0; position < K1_BOARD_CELLS; ++position) {
        out_board[position] = board[position];
    }
    for (lane = 0; lane < K1_BOARD_SIDE; ++lane) {
        int32_t cells[K1_BOARD_SIDE];
        int moved_into[K1_BOARD_SIDE] = {0, 0, 0, 0};
        int merged_into[K1_BOARD_SIDE] = {0, 0, 0, 0};
        int changed = 0;
        for (position = 0; position < K1_BOARD_SIDE; ++position) {
            cells[position] = board[k1_line_index(action, lane, position)];
        }
        for (position = 1; position < K1_BOARD_SIDE; ++position) {
            int32_t value = cells[position];
            if (value == 0) {
                continue;
            }
            if (cells[position - 1] == 0) {
                cells[position - 1] = value;
                cells[position] = 0;
                moved_into[position - 1] = 1;
                merged_into[position - 1] = 0;
                changed = 1;
            } else if (
                k1_can_merge(cells[position - 1], value) &&
                !moved_into[position - 1] &&
                !merged_into[position - 1]
            ) {
                cells[position - 1] = k1_merge_value(
                    cells[position - 1],
                    value
                );
                cells[position] = 0;
                moved_into[position - 1] = 0;
                merged_into[position - 1] = 1;
                changed = 1;
            }
        }
        for (position = 0; position < K1_BOARD_SIDE; ++position) {
            out_board[k1_line_index(action, lane, position)] = cells[position];
        }
        if (changed) {
            int row = 0;
            int col = 0;
            if (action == 0) {
                row = 3;
                col = lane;
            } else if (action == 1) {
                row = 0;
                col = lane;
            } else if (action == 2) {
                row = lane;
                col = 3;
            } else {
                row = lane;
                col = 0;
            }
            if (out_board[row * K1_BOARD_SIDE + col] == 0) {
                out_eligible_pairs[2 * eligible_count] = row;
                out_eligible_pairs[2 * eligible_count + 1] = col;
                eligible_count += 1;
            }
        }
    }
    *out_eligible_count = eligible_count;
    return 0;
}

static int k1_score_tile(int32_t value, int64_t *out_score) {
    int32_t quotient;
    int64_t score;
    if (value == 0 || value == 1 || value == 2) {
        *out_score = 0;
        return 0;
    }
    if (value < 3 || value % 3 != 0) {
        return 1;
    }
    quotient = value / 3;
    score = 3;
    while (quotient > 1 && quotient % 2 == 0) {
        quotient /= 2;
        score *= 3;
    }
    if (quotient != 1) {
        return 1;
    }
    *out_score = score;
    return 0;
}

int k1_score_board(const int32_t *board, int64_t *out_score) {
    int cell;
    int64_t total = 0;
    if (board == NULL || out_score == NULL) {
        return 1;
    }
    for (cell = 0; cell < K1_BOARD_CELLS; ++cell) {
        int64_t tile_score = 0;
        if (k1_score_tile(board[cell], &tile_score) != 0) {
            return 2;
        }
        total += tile_score;
    }
    *out_score = total;
    return 0;
}

static int k1_phase_index(const int32_t *board) {
    int cell;
    int remove_index = -1;
    int32_t built_max = 0;
    if (board[0] == K1_STARTER_TILE) {
        remove_index = 0;
    } else {
        for (cell = 0; cell < K1_BOARD_CELLS; ++cell) {
            if (board[cell] == K1_STARTER_TILE) {
                remove_index = cell;
                break;
            }
        }
    }
    for (cell = 0; cell < K1_BOARD_CELLS; ++cell) {
        if (cell != remove_index && board[cell] > built_max) {
            built_max = board[cell];
        }
    }
    if (built_max < 384) {
        return 0;
    }
    if (built_max < 1536) {
        return 1;
    }
    if (built_max < 3072) {
        return 2;
    }
    return 3;
}

int k1_eval_composite(
    const int32_t *boards,
    size_t board_count,
    const int16_t *rank_lut,
    size_t rank_lut_length,
    const int8_t *pattern_lengths,
    const int8_t *pattern_cells,
    const float *const *table_pointers,
    const int64_t *table_lengths,
    const double *phase_model_coefficients,
    double *out_values
) {
    size_t board_index;
    if (
        boards == NULL ||
        rank_lut == NULL ||
        pattern_lengths == NULL ||
        pattern_cells == NULL ||
        table_pointers == NULL ||
        table_lengths == NULL ||
        phase_model_coefficients == NULL ||
        out_values == NULL
    ) {
        return 1;
    }
    for (board_index = 0; board_index < board_count; ++board_index) {
        const int32_t *board = boards + board_index * K1_BOARD_CELLS;
        int16_t ranks[K1_BOARD_CELLS];
        double model_values[K1_MODELS] = {0.0, 0.0, 0.0, 0.0};
        int phase = k1_phase_index(board);
        int cell;
        int model;
        for (cell = 0; cell < K1_BOARD_CELLS; ++cell) {
            int32_t value = board[cell];
            if (value < 0 || (size_t)value >= rank_lut_length) {
                return 2;
            }
            ranks[cell] = rank_lut[value];
            if (ranks[cell] < 0 || ranks[cell] >= K1_NUM_RANKS) {
                return 3;
            }
        }
        for (model = 0; model < K1_MODELS; ++model) {
            double accumulator = 0.0;
            int symmetry;
            for (symmetry = 0; symmetry < K1_SYMMETRIES; ++symmetry) {
                int pattern;
                for (pattern = 0; pattern < K1_PATTERNS; ++pattern) {
                    int length = pattern_lengths[pattern];
                    int position;
                    int64_t table_index = 0;
                    int pointer_index = (
                        (model * K1_PHASES + phase) * K1_PATTERNS + pattern
                    );
                    const float *table = table_pointers[pointer_index];
                    if (length <= 0 || length > K1_MAX_PATTERN || table == NULL) {
                        return 4;
                    }
                    for (position = 0; position < length; ++position) {
                        int map_index = (
                            (pattern * K1_SYMMETRIES + symmetry) *
                            K1_MAX_PATTERN + position
                        );
                        int board_cell = pattern_cells[map_index];
                        if (board_cell < 0 || board_cell >= K1_BOARD_CELLS) {
                            return 5;
                        }
                        table_index = (
                            table_index * K1_NUM_RANKS + ranks[board_cell]
                        );
                    }
                    if (
                        table_index < 0 ||
                        table_index >= table_lengths[pointer_index]
                    ) {
                        return 6;
                    }
                    accumulator += (double)table[table_index];
                }
            }
            model_values[model] = accumulator;
        }
        {
            const double *coefficients = (
                phase_model_coefficients + phase * K1_MODELS
            );
            double value = coefficients[0] * model_values[0];
            value += coefficients[1] * model_values[1];
            value += coefficients[2] * model_values[2];
            value += coefficients[3] * model_values[3];
            out_values[board_index] = value;
        }
    }
    return 0;
}

int k1_post_spawn_rows(
    const int32_t *board,
    const int16_t *rank_lut,
    size_t rank_lut_length,
    const int8_t *pattern_lengths,
    const int8_t *pattern_cells,
    const float *const *table_pointers,
    const int64_t *table_lengths,
    const double *phase_model_coefficients,
    int32_t *out_boards,
    int32_t *out_eligible_pairs,
    int32_t *out_eligible_counts,
    int32_t *out_legal_flags,
    int64_t *out_before_score,
    int64_t *out_after_scores,
    double *out_leaf_values
) {
    int action;
    int64_t before_score = 0;
    if (
        board == NULL ||
        out_boards == NULL ||
        out_eligible_pairs == NULL ||
        out_eligible_counts == NULL ||
        out_legal_flags == NULL ||
        out_before_score == NULL ||
        out_after_scores == NULL ||
        out_leaf_values == NULL
    ) {
        return 1;
    }
    if (k1_score_board(board, &before_score) != 0) {
        return 2;
    }
    *out_before_score = before_score;
    for (action = 0; action < K1_ACTIONS; ++action) {
        int32_t count = 0;
        int code = k1_base_move(
            board,
            action,
            out_boards + action * K1_BOARD_CELLS,
            out_eligible_pairs + action * K1_BOARD_SIDE * 2,
            &count
        );
        if (code != 0) {
            return 10 + code;
        }
        out_eligible_counts[action] = count;
        out_legal_flags[action] = count > 0 ? 1 : 0;
        out_after_scores[action] = before_score;
        out_leaf_values[action] = 0.0;
        if (count > 0) {
            code = k1_score_board(
                out_boards + action * K1_BOARD_CELLS,
                out_after_scores + action
            );
            if (code != 0) {
                return 20 + code;
            }
            code = k1_eval_composite(
                out_boards + action * K1_BOARD_CELLS,
                1,
                rank_lut,
                rank_lut_length,
                pattern_lengths,
                pattern_cells,
                table_pointers,
                table_lengths,
                phase_model_coefficients,
                out_leaf_values + action
            );
            if (code != 0) {
                return 30 + code;
            }
        }
    }
    return 0;
}
