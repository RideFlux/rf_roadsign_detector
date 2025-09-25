
#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <optimizer.h>

class Matcher
{
public: 
  Matcher(const std::string& target_path, const std::string& source_path);
  void match();
  void saveResultPCD(const std::string& path);
  void visualizeResult(){visualizeRegistration(target_cloud_,source_cloud_,updated_cloud_);}

private:
  void findCorrespondences(pcl::PointCloud<Point_T>::Ptr in_cloud, pcl::PointCloud<Point_T>::Ptr ref_cloud);
  void visualizeRegistration(const pcl::PointCloud<Point_T>::ConstPtr& target,
    const pcl::PointCloud<Point_T>::ConstPtr& source,
    const pcl::PointCloud<Point_T>::ConstPtr& result);

public: 

private:
  const int MAX_ITER_NUM = 20;
  const double CONVERGE_THRESHOLD_TRANSLATION = 0.025;
  const double CONVERGE_THRESHOLD_ROTATION = 0.25/180 * M_PI;

  pcl::PointCloud<Point_T>::Ptr target_cloud_;
  pcl::PointCloud<Point_T>::Ptr source_cloud_;
  pcl::PointCloud<Point_T>::Ptr updated_cloud_;

  pcl::search::KdTree<Point_T>::Ptr kdtree_;

  Optimizer optimizer_;
  std::vector<Optimizer::Correspondence> correspondence_vec_;
};