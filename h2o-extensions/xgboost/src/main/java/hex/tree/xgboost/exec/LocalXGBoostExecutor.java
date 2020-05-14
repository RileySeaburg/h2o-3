package hex.tree.xgboost.exec;

import hex.DataInfo;
import hex.tree.xgboost.BoosterParms;
import hex.tree.xgboost.XGBoostModel;
import hex.tree.xgboost.XGBoostUtils;
import hex.tree.xgboost.matrix.FrameMatrixLoader;
import hex.tree.xgboost.matrix.MatrixLoader;
import hex.tree.xgboost.matrix.RemoteMatrixLoader;
import hex.tree.xgboost.predict.XGBoostVariableImportance;
import hex.tree.xgboost.rabit.RabitTrackerH2O;
import hex.tree.xgboost.task.XGBoostCleanupTask;
import hex.tree.xgboost.task.XGBoostSetupTask;
import hex.tree.xgboost.task.XGBoostUpdateTask;
import hex.tree.xgboost.util.BoosterHelper;
import hex.tree.xgboost.util.FeatureScore;
import ml.dmlc.xgboost4j.java.*;
import org.apache.log4j.Logger;
import water.H2O;
import water.Key;
import water.fvec.Frame;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Files;
import java.util.HashMap;
import java.util.Map;

public class LocalXGBoostExecutor implements XGBoostExecutor {

    public final Key modelKey;
    private final RabitTrackerH2O rt;
    private final XGBoostSetupTask setupTask;
    
    private XGBoostUpdateTask updateTask;
    
    private byte[] latestBooster;
    
    /**
     * Used when executing from a remote model
     */
    public LocalXGBoostExecutor(Key key, XGBoostExecReq.Init init) {
        modelKey = key;
        rt = new RabitTrackerH2O(init.num_nodes);
        BoosterParms boosterParams = BoosterParms.fromMap(init.parms);
        boolean[] nodes = new boolean[H2O.CLOUD.size()];
        for (int i = 0; i < init.num_nodes; i++) nodes[i] = init.nodes[i] != null;
        MatrixLoader loader = new RemoteMatrixLoader(init.matrix_dir_path, init.nodes);
        setupTask = new XGBoostSetupTask(
            modelKey, null, boosterParams, init.checkpoint_bytes, getRabitEnv(), nodes, loader
        );
    }

    /**
     * Used when executing from a local model
     */
    public LocalXGBoostExecutor(XGBoostModel model, Frame train) {
        modelKey = model._key;
        XGBoostSetupTask.FrameNodes trainFrameNodes = XGBoostSetupTask.findFrameNodes(train);
        rt = new RabitTrackerH2O(trainFrameNodes.getNumNodes());
        byte[] checkpointBytes = null;
        if (model._parms.hasCheckpoint()) {
            checkpointBytes = model.model_info()._boosterBytes;
        }
        DataInfo dataInfo = model.model_info().dataInfo();
        BoosterParms boosterParms = XGBoostModel.createParams(model._parms, model._output.nclasses(), dataInfo.coefNames());
        model._output._native_parameters = boosterParms.toTwoDimTable();
        MatrixLoader loader = new FrameMatrixLoader(model, train);
        setupTask = new XGBoostSetupTask(
            modelKey, model._parms._save_matrix_directory, boosterParms, checkpointBytes, 
            getRabitEnv(), trainFrameNodes._nodes, loader
        );
    }
    
    @Override
    public byte[] setup() {
        startRabitTracker();
        setupTask.run();
        updateTask = new XGBoostUpdateTask(setupTask, 0).run();
        return updateTask.getBoosterBytes();
    }

    // Don't start the tracker for 1 node clouds -> the GPU plugin fails in such a case
    private void startRabitTracker() {
        if (H2O.CLOUD.size() > 1) {
            rt.start(0);
        }
    }

    private void stopRabitTracker() {
        if(H2O.CLOUD.size() > 1) {
            rt.waitFor(0);
            rt.stop();
        }
    }

    // XGBoost seems to manipulate its frames in case of a 1 node distributed version in a way the GPU plugin can't handle
    // Therefore don't use RabitTracker envs for 1 node
    private Map<String, String> getRabitEnv() {
        if(H2O.CLOUD.size() > 1) {
            return rt.getWorkerEnvs();
        } else {
            return new HashMap<>();
        }
    }

    @Override
    public void update(int treeId) {
        updateTask = new XGBoostUpdateTask(setupTask, treeId);
        updateTask.run();
    }

    @Override
    public byte[] updateBooster() {
        latestBooster = updateTask.getBoosterBytes();
        return latestBooster;
    }

    @Override
    public void cleanup() {
        XGBoostCleanupTask.cleanUp(setupTask);
        stopRabitTracker();
    }

}
