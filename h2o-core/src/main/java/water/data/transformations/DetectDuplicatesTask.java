package water.data.transformations;

import water.Iced;
import water.MRTask;
import water.fvec.Chunk;
import water.util.ArrayUtils;
import water.util.IcedDouble;

class DetectDuplicatesTask extends MRTask<DetectDuplicatesTask> {

    private final int[] comparedColumnIndices;
    private DuplicatedRow[] duplicatedRows;
    private RowSample[] firstRowSamples;
    private final int labelRowChunkId;
    private transient int equalRowsStartIndex = -1;

    DetectDuplicatesTask(final int[] comparedColumnIndices, final int labelRowChunkId) {
        this.comparedColumnIndices = comparedColumnIndices;
        this.duplicatedRows = new DuplicatedRow[0];
        this.firstRowSamples = new RowSample[1];
        this.labelRowChunkId = labelRowChunkId;
    }

    @Override
    public void map(Chunk[] chunks) {
        final Iced[] firstRowCells = getRowSample(chunks, 0);
        this.firstRowSamples[0] = new RowSample(firstRowCells, chunks[0].cidx(), chunks[labelRowChunkId].at8(0));

        long maxRowLabel = -1;
        for (int row = 0; row < chunks[0].len() - 1; row++) {
            boolean rowsAreEqual = true;

            for (int column : comparedColumnIndices) {
                rowsAreEqual = compareColumnValues(chunks[column], row);
                if (!rowsAreEqual) break;
            }

            if (rowsAreEqual && firstDuplicateRow()) {
                equalRowsStartIndex = row;
                final long rowLabel = chunks[labelRowChunkId].at8(row);
                if (rowLabel > maxRowLabel) maxRowLabel = rowLabel;
                if (row == chunks[0].len() - 2) {
                    createDuplicatedRowRecord(chunks, row, rowsAreEqual, maxRowLabel);
                    maxRowLabel = -1;
                }
            } else if ((!rowsAreEqual && !firstDuplicateRow()) || row == chunks[0].len() - 2) {
                final long rowLabel = chunks[labelRowChunkId].at8(row);
                if (rowLabel > maxRowLabel) maxRowLabel = rowLabel;
                createDuplicatedRowRecord(chunks, row, rowsAreEqual, maxRowLabel);
                maxRowLabel = -1;
            }
        }
    }

    private void createDuplicatedRowRecord(final Chunk[] chunks, final int row, final boolean rowsAreEqual, long maxRowLabel) {
        final boolean reachesChunkEnd = row == chunks[0].len() - 2 && rowsAreEqual;
        final int endIndex = reachesChunkEnd ? row + 1 : row;
        final DuplicatedRow duplicatedRow;
        if (reachesChunkEnd) {
            final Iced[] lastRowCells = getRowSample(chunks, chunks[0].len() - 1);
            final RowSample lastRowSample = new RowSample(lastRowCells, chunks[0].cidx(), chunks[labelRowChunkId].at8(row));

            duplicatedRow = new DuplicatedRow(chunks[0].cidx(), equalRowsStartIndex, endIndex, maxRowLabel,
                    lastRowSample);
        } else {
            duplicatedRow = new DuplicatedRow(chunks[0].cidx(), equalRowsStartIndex, endIndex, maxRowLabel);
        }
        this.duplicatedRows = ArrayUtils.append(this.duplicatedRows, duplicatedRow);
        equalRowsStartIndex = -1;
    }

    private Iced[] getRowSample(Chunk[] chunks, final int row) {
        final Iced[] rowCells = new Iced[comparedColumnIndices.length];
        int firstRowCellsIndex = 0;
        for (final int column : comparedColumnIndices) {
            rowCells[firstRowCellsIndex] = new IcedDouble(chunks[column].at8(row));
            firstRowCellsIndex++;
        }
        return rowCells;
    }


    private static boolean compareColumnValues(final Chunk chunk, final int row) {
        final double currentRow = chunk.atd(row);
        final double nextRow = chunk.atd(row + 1);

        return currentRow == nextRow;
    }

    @Override
    public void reduce(DetectDuplicatesTask mrt) {
        this.duplicatedRows = ArrayUtils.append(this.duplicatedRows, mrt.duplicatedRows);
        this.firstRowSamples = ArrayUtils.append(this.firstRowSamples, mrt.firstRowSamples);
    }

    private final boolean firstDuplicateRow() {
        return equalRowsStartIndex == -1;
    }

    public DuplicatedRow[] getDuplicatedRows() {
        return duplicatedRows;
    }

    public RowSample[] getFirstRowSamples() {
        return firstRowSamples;
    }
}
