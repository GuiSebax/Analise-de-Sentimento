package jgam.ai;

import jgam.game.BoardSetup;
import jgam.game.PossibleMoves;
import jgam.game.SingleMove;

import java.util.List;

/**
 * Implementação de IA utilizando o algoritmo Minimax com poda Alfa-Beta.
 * Guilherme Frare Clemente - RA:124349
 * Marcos Vinicius de Oliveira - RA:124408
 */
public class IAdoGuilhermeEMarcosMM implements AI {
    private static final int MAX_DEPTH = 1; // profundidade da busca

    @Override
    public void init() throws Exception {
        
    }

    @Override
    public void dispose() {
       
    }

    @Override
    public String getName() {
        return "Minimax Poda Alfa-Beta";
    }

    @Override
    public String getDescription() {
        return "IA implementada pela equipe Marcos e Guilherme";
    }

    private double heuristic(BoardSetup bs) {
        double evaluation = 0.0;

        int player = bs.getPlayerAtMove();

        for (int i = 1; i < 25; i++) {
            int numCheckers = bs.getPoint(player, i);
            switch (numCheckers) {
                case 2:
                    evaluation += 100.0;
                    break;
                case 1:
                    evaluation -= 75.0;
                    break;
                default:
                    evaluation -= 25.0 * numCheckers;
                    break;
            }
        }

        int totalPoints = bs.getPoint(player, 25);
        evaluation += 50.0 * totalPoints;

        return evaluation;
    }

    @Override
    public SingleMove[] makeMoves(BoardSetup bs) throws CannotDecideException {
        MoveEvaluation bestMoveEval = minimax(bs, MAX_DEPTH, Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY, true);
        if (bestMoveEval == null || bestMoveEval.getMoves() == null) {
            // Retornar um array vazio se não houver movimentos válidos
            return new SingleMove[0];
        }
        return bestMoveEval.getMoves();
    }

    @Override
    public int rollOrDouble(BoardSetup boardSetup) throws CannotDecideException {
        // Decidir se deve rolar ou dobrar
        double currentHeuristic = heuristic(boardSetup);
        if (currentHeuristic > 500) {
            return AI.DOUBLE; // dobrar se a vantagem for significativa
        }
        return AI.ROLL; // caso contrário, apenas rolar
    }

    @Override
    public int takeOrDrop(BoardSetup boardSetup) throws CannotDecideException {
        // Decidir se deve aceitar ou recusar uma oferta de dobrar
        double currentHeuristic = heuristic(boardSetup);
        if (currentHeuristic < -500) {
            return AI.DROP; // recusar se a situação estiver muito desfavorável
        }
        return AI.TAKE; // aceitar caso contrário
    }

    private MoveEvaluation minimax(BoardSetup boardSetup, int depth, double alpha, double beta, boolean maximizingPlayer) throws CannotDecideException {
        if (depth == 0 || boardSetup.isGameOver()) {
            return new MoveEvaluation(heuristic(boardSetup), null);
        }

        PossibleMoves possibleMoves = new PossibleMoves(boardSetup);
        List<BoardSetup> moveList = possibleMoves.getPossbibleNextSetups();
        if (moveList == null || moveList.isEmpty()) {
            // Retornar uma avaliação nula se não houver movimentos possíveis
            return new MoveEvaluation(heuristic(boardSetup), null);
        }

        MoveEvaluation bestMoveEval = new MoveEvaluation(maximizingPlayer ? Double.NEGATIVE_INFINITY : Double.POSITIVE_INFINITY, null);

        for (int i = 0; i < moveList.size(); i++) {
            BoardSetup nextBoardSetup = moveList.get(i);
            MoveEvaluation eval = minimax(nextBoardSetup, depth - 1, alpha, beta, !maximizingPlayer);
            if (maximizingPlayer) {
                if (eval.getEvaluation() > bestMoveEval.getEvaluation()) {
                    bestMoveEval = new MoveEvaluation(eval.getEvaluation(), possibleMoves.getMoveChain(i));
                }
                alpha = Math.max(alpha, eval.getEvaluation());
            } else {
                if (eval.getEvaluation() < bestMoveEval.getEvaluation()) {
                    bestMoveEval = new MoveEvaluation(eval.getEvaluation(), possibleMoves.getMoveChain(i));
                }
                beta = Math.min(beta, eval.getEvaluation());
            }
            if (beta <= alpha) {
                break;
            }
        }

        return bestMoveEval;
    }

    private static class MoveEvaluation {
        private final double evaluation;
        private final SingleMove[] moves;

        public MoveEvaluation(double evaluation, SingleMove[] moves) {
            this.evaluation = evaluation;
            this.moves = moves;
        }

        public double getEvaluation() {
            return evaluation;
        }

        public SingleMove[] getMoves() {
            return moves;
        }
    }
}
