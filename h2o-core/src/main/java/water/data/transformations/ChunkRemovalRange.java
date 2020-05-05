package water.data.transformations;

import water.fvec.Chunk;
import water.fvec.NewChunk;

public class ChunkRemovalRange {

    private final int startRow;
    private final int endRow;

    public ChunkRemovalRange(int startRow, int endRow) {
        this.startRow = startRow;
        this.endRow = endRow;
    }
    
    public void apply(final Chunk originalChunk, final NewChunk newChunk){
        
    }
}
