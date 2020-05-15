package hex.tree.xgboost;

import hex.Model;
import hex.ModelBuilder;
import hex.ScoreKeeper;
import hex.glm.GLMModel;
import hex.tree.PlattScalingHelper;
import water.util.IcedHashMap;
import water.util.TwoDimTable;

import java.util.ArrayList;
import java.util.List;

public class XGBoostOutput extends Model.Output implements Model.GetNTrees, PlattScalingHelper.OutputWithCalibration {
  public XGBoostOutput(XGBoost b) {
    super(b);
    _scored_train = new ScoreKeeper[]{new ScoreKeeper(Double.NaN)};
    _scored_valid = new ScoreKeeper[]{new ScoreKeeper(Double.NaN)};
  }

  int _nums;
  int _cats;
  int[] _catOffsets;
  boolean _useAllFactorLevels;
  public boolean _sparse;

  public int _ntrees;
  public ScoreKeeper[/*ntrees+1*/] _scored_train;
  public ScoreKeeper[/*ntrees+1*/] _scored_valid;
  public ScoreKeeper[] scoreKeepers() {
    List<ScoreKeeper> skl = new ArrayList<>();
    ScoreKeeper[] ska = _validation_metrics != null ? _scored_valid : _scored_train;
    for( ScoreKeeper sk : ska )
      if (!sk.isEmpty())
        skl.add(sk);
    return skl.toArray(new ScoreKeeper[0]);
  }
  public long[/*ntrees+1*/] _training_time_ms = {System.currentTimeMillis()};
  public TwoDimTable _variable_importances; // gain
  public TwoDimTable _variable_importances_cover;
  public TwoDimTable _variable_importances_frequency;
  public XgbVarImp _varimp;
  public TwoDimTable _native_parameters;

  public GLMModel _calib_model;

  @Override
  public TwoDimTable createInputFramesInformationTable(ModelBuilder modelBuilder) {
    List<String> colHeaders = new ArrayList<>();
    List<String> colTypes = new ArrayList<>();
    List<String> colFormat = new ArrayList<>();
    XGBoostModel.XGBoostParameters params = (XGBoostModel.XGBoostParameters) modelBuilder._parms;

    colHeaders.add("Input Frame"); colTypes.add("string"); colFormat.add("%s");
    colHeaders.add("Checksum"); colTypes.add("long"); colFormat.add("%d");

    final int rows = 3;
    TwoDimTable table = new TwoDimTable(
            "Input Frames Information", null,
            new String[rows],
            colHeaders.toArray(new String[0]),
            colTypes.toArray(new String[0]),
            colFormat.toArray(new String[0]),
            "");

    table.set(0, 0, "training_frame");
    table.set(1, 0, "validation_frame");
    table.set(2, 0, "calibration_frame");
    table.set(0, 1, modelBuilder.train().checksum());
    table.set(1, 1, params._valid != null ? modelBuilder.valid().checksum() : -1);
    table.set(2, 1, params.getCalibrationFrame() != null ? params.getCalibrationFrame().checksum() : -1);

    return table;
  }
  
  @Override
  public int getNTrees() {
    return _ntrees;
  }

  @Override
  public GLMModel calibrationModel() {
    return _calib_model;
  }
}
