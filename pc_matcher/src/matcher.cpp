#include <matcher.h>

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

void Matcher::match(){
  // if(optimizer_.isSourceCloudEmpty() || optimizer_.isTargetCloudEmpty()) return;
  
  Eigen::Affine3f current_transform = Eigen::Affine3f::Identity();
  pcl::PointCloud<Point_T>::Ptr updated_cloud(new pcl::PointCloud<Point_T>());
  optimizer_.setTargetCloud(target_cloud_);

  for(int iter=0; iter<MAX_ITER_NUM; iter++){
    pcl::transformPointCloud(*source_cloud_, *updated_cloud, current_transform);

    findCorrespondences(updated_cloud,target_cloud_);
    optimizer_.setSourceCloud(updated_cloud);
    optimizer_.setCorrespondence(correspondence_vec_);
    optimizer_.optimize();
    auto update = optimizer_.getResult();
    optimizer_.clear();

    current_transform = update * current_transform;

    auto delta_t = update.translation().norm();
    auto delta_R = Eigen::AngleAxisf(update.rotation()).angle();

    if(delta_R < CONVERGE_THRESHOLD_ROTATION && delta_t < CONVERGE_THRESHOLD_TRANSLATION)  
      break;
  }
  std::cout << "result : " << current_transform.matrix() << std::endl;
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
    // Point_T closest_target_point = target_cloud_->points[ref_idx];

    correspondence_vec_.emplace_back(Optimizer::Correspondence(in_idx,ref_idx));
  }
}

void Matcher::saveResultPCD(const std::string& path){

  if (pcl::io::savePCDFileASCII(path, *updated_cloud_) == -1) {
    std::cerr << "cannot save result pcd " << path << std::endl;
  }
}

void Matcher::visualizeRegistration(
  const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& target,
  const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& source,
  const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& result) {
  
  pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer(" "));
  viewer->setBackgroundColor(0.1, 0.1, 0.1); // 배경 어둡게

  // Target 포인트 클라우드 (흰색)
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> target_color(target, 255, 255, 255);
  viewer->addPointCloud<pcl::PointXYZ>(target, target_color, "target cloud");
  viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.5, "target cloud");

  // Source 포인트 클라우드 (빨간색)
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> source_color(source, 255, 0, 0);
  viewer->addPointCloud<pcl::PointXYZ>(source, source_color, "source cloud");
  viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.5, "source cloud");

  // 정합된  포인트 클라우드 (녹색)
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> aligned_color(result, 0, 255, 0);
  viewer->addPointCloud<pcl::PointXYZ>(result, aligned_color, "aligned cloud");
  viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1.5, "aligned cloud");

  viewer->addCoordinateSystem(1.0);
  viewer->initCameraParameters();
  
  // 창이 닫힐 때까지 대기
  while (!viewer->wasStopped()) {
    viewer->spinOnce(100);
  }
}