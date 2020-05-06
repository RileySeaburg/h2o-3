package hex.tree.xgboost.exec;

public interface XGBoostExecutor {

    byte[] setup();

    void update(int treeId);

    byte[] updateBooster();

    void cleanup();

}
