#include <pcl/common/pca.h>
#include <pcl/common/transforms.h>

#include <matcher.h>
#include <iostream> 

Matcher::Matcher(const std::string& target_path, const std::string& source_path):
  target_cloud_(new pcl::PointCloud<Point_T>()),
  source_cloud_(new pcl::PointCloud<Point_T>()),
  updated_cloud_(new pcl::PointCloud<Point_T>()),
  kdtree_(new pcl::search::KdTree<Point_T>())
{
   if (pcl::io::loadPCDFile(target_path, *target_cloud_)) {
    std::cerr << "failed to open " << target_path << std::endl;
  }
  if (pcl::io::loadPCDFile(source_path, *source_cloud_)) {
    std::cerr << "failed to open " << source_path << std::endl;
  } 

  if(!target_cloud_ || target_cloud_->points.empty()){
    std::cerr << "target cloud is empty " << std::endl;
  }

  if(!source_cloud_ || source_cloud_->points.empty()){
    std::cerr << "source cloud is empty " << std::endl;
  }
}

void Matcher::match(bool visualize_process){
  std::cout << "Computing PCA-based initial alignment..." << std::endl;

  pcl::PCA<Point_T> pca_src;
  pca_src.setInputCloud(source_cloud_);
  Eigen::Matrix3f axes_src = pca_src.getEigenVectors();
  Eigen::Vector4f mean_src = pca_src.getMean();

  pcl::PCA<Point_T> pca_tgt;
  pca_tgt.setInputCloud(target_cloud_);
  Eigen::Matrix3f axes_tgt = pca_tgt.getEigenVectors();
  Eigen::Vector4f mean_tgt = pca_tgt.getMean();

  Eigen::Matrix3f R = axes_tgt * axes_src.transpose();

  if (R.determinant() < 0) {
    R.col(2) *= -1;
  }

  Eigen::Vector3f t = mean_tgt.head<3>() - R * mean_src.head<3>();

  Eigen::Matrix4f T_init = Eigen::Matrix4f::Identity();
  T_init.block<3, 3>(0, 0) = R; 
  T_init.block<3, 1>(0, 3) = t; 
  
  Eigen::Affine3f current_transform(T_init); 

  std::cout << "Initial alignment computed. Starting ICP loop..." << std::endl;

  pcl::PointCloud<Point_T>::Ptr updated_cloud(new pcl::PointCloud<Point_T>());
  optimizer_.setTargetCloud(target_cloud_);

  for(int iter=0; iter<MAX_ITER_NUM; iter++){
    if (iter == 0) {
        pcl::transformPointCloud(*source_cloud_, *updated_cloud, current_transform);
    } 

    findCorrespondences(updated_cloud,target_cloud_);
    optimizer_.setSourceCloud(updated_cloud);
    optimizer_.setCorrespondence(correspondence_vec_);
    optimizer_.optimize();
    auto update = optimizer_.getResult();
    optimizer_.clear();

    if(visualize_process){
      static pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer(" "));
      viewer->addCoordinateSystem(1.0);
      viewer->initCameraParameters();
      viewer->setCameraPosition(source_cloud_->points[0].x + 5.0 ,source_cloud_->points[0].y - 5.0, 0, 
                                target_cloud_->points[0].x + 5.0 ,target_cloud_->points[0].y - 5.0, 0, 
                                0,0,1);

      visualizeRegistration(viewer, target_cloud_, updated_cloud, updated_cloud, correspondence_vec_);
    }

    current_transform = update * current_transform;

    pcl::transformPointCloud(*source_cloud_, *updated_cloud, current_transform);

    auto delta_t = update.translation().norm();
    auto delta_R = Eigen::AngleAxisf(update.rotation()).angle();

    if(delta_R < CONVERGE_THRESHOLD_ROTATION && delta_t < CONVERGE_THRESHOLD_TRANSLATION) {
      std::cout << "Converged after " << iter + 1 << " iterations." << std::endl;
      break;
    }
  }
  *updated_cloud_ = *updated_cloud;
}

void Matcher::findCorrespondences(pcl::PointCloud<Point_T>::Ptr in_cloud, pcl::PointCloud<Point_T>::Ptr ref_cloud){
  correspondence_vec_.clear();
  correspondence_vec_.reserve(in_cloud->points.size());

  kdtree_->setInputCloud(ref_cloud);

  std::vector<int> idx_vec;
  std::vector<float> dist_vec;

  for(std::size_t in_idx=0; in_idx<in_cloud->points.size(); in_idx++){
    const auto& query = in_cloud->points[in_idx];
    kdtree_->nearestKSearch(query, 1, idx_vec,dist_vec);
    const int ref_idx = idx_vec[0];

    correspondence_vec_.emplace_back(Optimizer::Correspondence(in_idx,ref_idx));
  }
}

void Matcher::saveResultPCD(const std::string& path){

  if (pcl::io::savePCDFileASCII(path, *updated_cloud_) == -1) {
    std::cerr << "cannot save result pcd " << path << std::endl;
  }
}
void Matcher::visualizeResult(){
  pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Result"));
  viewer->addCoordinateSystem(1.0);
  viewer->initCameraParameters();
  int mid_idx = target_cloud_->points.size()/2;
  viewer->setCameraPosition(target_cloud_->points[mid_idx].x + 5.0 ,target_cloud_->points[mid_idx].y - 5.0, target_cloud_->points[mid_idx].z, 
                                target_cloud_->points[mid_idx].x ,target_cloud_->points[mid_idx].y, target_cloud_->points[mid_idx].z, 
                                0,0,1);
  visualizeRegistration(viewer,target_cloud_,source_cloud_,updated_cloud_);
  std::cout << "If target cloud is not visible, please click the viewer window and scroll your mouse wheel once." << std::endl;
  while (!viewer->wasStopped()) {
    viewer->spinOnce(100);
  }
}

void Matcher::visualizeRegistration(
  const pcl::visualization::PCLVisualizer::Ptr& viewer,
  const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& target,
  const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& source,
  const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& result,
  const std::vector<Optimizer::Correspondence>& correspondence_vec) {
  
  viewer->removeAllPointClouds();
  viewer->removeAllShapes();
  viewer->setBackgroundColor(0.1, 0.1, 0.1); 

  // Target 포인트 클라우드 (흰색)
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> target_color(target, 255, 255, 255);
  viewer->addPointCloud<pcl::PointXYZ>(target, target_color, "target cloud");
  viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.0, "target cloud");
  viewer->addText("WHITE : Target Cloud", 10, 50, 16, 1.0, 1.0, 1.0, "target_text");

  // Source 포인트 클라우드 (빨간색)
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_color(source, 255, 0, 0);
  viewer->addPointCloud<pcl::PointXYZ>(source, source_color, "source cloud");
  viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.0, "source cloud");
  viewer->addText("   RED   : Source Cloud", 10, 70, 16, 1.0, 0.0, 0.0, "source_text");

  // 정합된  포인트 클라우드 (녹색)
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> aligned_color(result, 0, 255, 0);
  viewer->addPointCloud<pcl::PointXYZ>(result, aligned_color, "aligned cloud");
  viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.5, "aligned cloud");
  viewer->addText("GREEN : Result Cloud", 10, 90, 16, 0.0, 1.0, 0.0, "result_text");

  if(!correspondence_vec.empty()){
    const int max_lines_to_show = 500;
    const double step = static_cast<double>(correspondence_vec.size()) / max_lines_to_show;

    for(double i=0; i < correspondence_vec.size(); i += step){
      const auto& corr = correspondence_vec[static_cast<size_t>(i)];
      const auto& src = result->points[corr.in_idx];
      const auto& tgt = target->points[corr.ref_idx];

      std::string line_id = "line" + std::to_string(i);
      viewer->addLine<Point_T>(src, tgt, 255, 255, 0, line_id); // 노란색 라인
    }
  }

  viewer->spinOnce(100);
}