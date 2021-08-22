CREATE TABLE `file_index` (
  `file_id` int NOT NULL AUTO_INCREMENT,
  `original_file` mediumtext NOT NULL,
  `image_extracted` tinyint NOT NULL,
  `text_extracted` tinyint NOT NULL,
  `tika_status` varchar(50) DEFAULT NULL,
  `ocr` tinyint DEFAULT NULL,
  `image_dir_ref` mediumtext,
  `text_json_ref` mediumtext,
  `ext` varchar(45) DEFAULT NULL,
  `error` mediumtext,
  `label` mediumtext,
  PRIMARY KEY (`file_id`)
) ENGINE=InnoDB AUTO_INCREMENT=11342 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
